"""
EMCAF-Net (nucleinet): Edge-aware Multi-Scale Contextual Attention Fusion Network
for binary nuclei segmentation.

Architecture:
  Encoder : PVTv2-B2  (E1–E4, dims [64,128,320,512])
  EMF     : Edge-aware Multi-scale Fusion  (nsb1–nsb4)
  CAF/AFB : Contextual Attention Fusion    (afb1–afb4)
  Decoder : Depthwise-separable conv + transposed conv  (dec1–dec4, dec11)
  Head    : 1×1 conv → binary logit  (outc)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running directly: python model/nucleinet.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.pvtv2 import pvt_v2_b2


# ── Channel Attention (inside EMF) ───────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Channel recalibration via global average pooling (paper Eq. 12).
    Squeeze: AdaptiveAvgPool2d → two FC (1×1 conv) layers → Sigmoid.
    """
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)          # global average pooling per Eq. 12
        self.fc1      = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))


# ── EMF: Edge-aware Multi-scale Fusion ───────────────────────────────────────
class EMF(nn.Module):
    """
    Edge-aware Multi-scale Fusion block (paper Section 3.2.2, Eq. 12-19).
      1. Channel recalibration (ChannelAttention)
      2. Local-contrast edge gate:  edge = X − AvgPool(X)
         weight = Sigmoid(BN(Conv(edge)))
         enhanced = weight * X + X   → PReLU
      3. Parallel standard conv (3×3, d=1) + dilated conv (3×3, d=3)
         output = standard + dilated + enhanced
    """
    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)

        # edge gate
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv_gate = nn.Conv2d(in_channel, in_channel, 1)
        self.bn_gate   = nn.BatchNorm2d(in_channel)
        self.sigmoid   = nn.Sigmoid()
        self.prelu     = nn.PReLU(in_channel)

        # parallel convolutions
        self.conv_std    = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.conv_dilate = nn.Conv2d(in_channel, in_channel, 3, padding=3, dilation=3)

    def forward(self, x):
        x = self.channel_attention(x) * x

        edge     = x - self.avg_pool(x)
        weight   = self.sigmoid(self.bn_gate(self.conv_gate(edge)))
        enhanced = self.prelu(weight * x + x)

        out = self.conv_std(x) + self.conv_dilate(x)
        return out + enhanced


# ── Decoder block ─────────────────────────────────────────────────────────────
def _dw_conv(inp, oup):
    """Depthwise-separable convolution block."""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def decoder(inp, oup):
    """Depthwise-separable conv followed by ×2 transposed conv upsample."""
    return nn.Sequential(
        _dw_conv(inp, oup),
        nn.ConvTranspose2d(oup, oup, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


# ── AFB: Contextual Attention Fusion (= CAF in paper) ─────────────────────────
class AFB(nn.Module):
    """
    Contextual Attention Fusion block (paper Section 3.2.3, Eq. 20-24).
    Two independent global avg-pool → Conv→BN→ReLU→Conv→BN→Sigmoid branches
    produce attention weights; output = high_feat * w_high + low_feat * w_low.
    """
    def __init__(self, mid_ch, out_ch):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_high = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.Sigmoid())
        self.branch_low = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.Sigmoid())

    def forward(self, input_high, input_low):
        w_high = self.branch_high(self.global_pool(input_high))
        w_low  = self.branch_low( self.global_pool(input_low))
        return input_high * w_high + input_low * w_low


# ── EMCAF-Net ─────────────────────────────────────────────────────────────────
class nucleinet(nn.Module):
    """
    EMCAF-Net.

    Forward flow (E = encoder, nsb = EMF, afb = CAF/AFB, dec = decoder):
      E4 → nsb4(c4) → afb4(c4, x4)  [stage-4: self-refine, no prior decoder] → dec4
      E3 → nsb3(c3) → afb3(dec4, c3) → dec3
      E2 → nsb2(c2) → afb2(dec3, c2) → dec2
      E1 → nsb1(c1) → afb1(dec2, c1) → dec1 → dec11 → outc → binary logit
    """
    def __init__(self):
        super().__init__()

        # Encoder: PVTv2-B2  (embed_dims = [64, 128, 320, 512])
        self.backbone = pvt_v2_b2()
        pretrained_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pvt_v2_b2.pth')
        if os.path.exists(pretrained_path):
            save_model  = torch.load(pretrained_path, map_location='cpu')
            model_dict  = self.backbone.state_dict()
            state_dict  = {k: v for k, v in save_model.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            print(f'[nucleinet] Loaded PVTv2-B2 weights from {pretrained_path} '
                  f'({len(state_dict)}/{len(model_dict)} backbone tensors matched)')
            if len(state_dict) < 0.9 * len(model_dict):
                print('[nucleinet] WARNING: fewer than 90% of backbone weights matched the '
                      'pretrained file. Check pvt_v2_b2.pth.')
        else:
            print(f'[nucleinet] WARNING: pretrained weights not found at {pretrained_path}. '
                  'Training from scratch.')

        # EMF blocks (one per encoder stage)
        self.nsb4 = EMF(512)
        self.nsb3 = EMF(320)
        self.nsb2 = EMF(128)
        self.nsb1 = EMF(64)

        # CAF/AFB blocks
        self.afb4 = AFB(512, 512)
        self.afb3 = AFB(320, 320)
        self.afb2 = AFB(128, 128)
        self.afb1 = AFB(64,  64)

        # Decoder blocks
        self.dec4  = decoder(512, 320)
        self.dec3  = decoder(320, 128)
        self.dec2  = decoder(128, 64)
        self.dec1  = decoder(64,  16)
        self.dec11 = decoder(16,  8)

        # Segmentation head
        self.outc = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        pvt = self.backbone(x)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]
        # x1: [B, 64,  H/4,  W/4]
        # x2: [B, 128, H/8,  W/8]
        # x3: [B, 320, H/16, W/16]
        # x4: [B, 512, H/32, W/32]

        # Stage 4 (deepest): no prior decoder output → AFB fuses EMF(x4) with raw x4
        c4   = self.nsb4(x4)
        d4   = self.dec4(self.afb4(c4, x4))

        # Stage 3
        c3   = self.nsb3(x3)
        d3   = self.dec3(self.afb3(d4, c3))

        # Stage 2
        c2   = self.nsb2(x2)
        d2   = self.dec2(self.afb2(d3, c2))

        # Stage 1
        c1   = self.nsb1(x1)
        d1   = self.dec1(self.afb1(d2, c1))

        out  = self.dec11(d1)
        return self.outc(out)   # raw logit; apply sigmoid at inference / use BCEWithLogitsLoss


# Public alias: the paper calls this model EMCAF-Net. The class name stays
# `nucleinet` so existing checkpoints load without renaming state-dict keys.
EMCAFNet = nucleinet


# ── Profiling (run: python -m model.nucleinet) ────────────────────────────────────
if __name__ == '__main__':
    from thop import profile

    model = nucleinet()
    model.eval()
    dummy = torch.randn(1, 3, 512, 512)

    macs, params = profile(model, inputs=(dummy,), verbose=False)
    print(f'Params : {params / 1e6:.3f} M')
    # thop returns multiply-accumulate operations, not FLOPs (1 MAC = 2 FLOPs)
    print(f'GMACs  : {macs / 1e9:.3f}   (= {2 * macs / 1e9:.3f} GFLOPs)')
