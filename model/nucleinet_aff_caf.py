"""
Ablation Variant 5: EMCAF-Net with AFF (Attentional Feature Fusion) replacing CAF/AFB.
Reference: Dai et al., "Attentional Feature Fusion", WACV 2021.
AFF fuses two feature maps using a Multi-Scale Channel Attention (MSCA) module
that combines both global (GAP-based) and local (point-wise conv) attention branches.
EMF block is unchanged. Only the fusion module is replaced.
"""
import os, sys, time
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.pvtv2 import pvt_v2_b2


# ── Channel Attention (inside EMF) -- unchanged ───────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))


# ── EMF -- unchanged ──────────────────────────────────────────────────────────
class EMF(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.avg_pool    = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv_gate   = nn.Conv2d(in_channel, in_channel, 1)
        self.bn_gate     = nn.BatchNorm2d(in_channel)
        self.sigmoid     = nn.Sigmoid()
        self.prelu       = nn.PReLU(in_channel)
        self.conv_std    = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.conv_dilate = nn.Conv2d(in_channel, in_channel, 3, padding=3, dilation=3)

    def forward(self, x):
        x        = self.channel_attention(x) * x
        edge     = x - self.avg_pool(x)
        weight   = self.sigmoid(self.bn_gate(self.conv_gate(edge)))
        enhanced = self.prelu(weight * x + x)
        out      = self.conv_std(x) + self.conv_dilate(x)
        return out + enhanced


# ── AFF: Attentional Feature Fusion (replaces CAF/AFB) ───────────────────────
class AFF(nn.Module):
    """
    Attentional Feature Fusion (Dai et al., WACV 2021).
    Fuses two inputs (high, low) via multi-scale channel attention:
      - Global branch: GAP -> Conv -> BN -> ReLU -> Conv -> BN
      - Local branch:  Conv(1x1) -> BN -> ReLU -> Conv(1x1) -> BN
      Combined attention: sigmoid(global + local)
      Output: high * w + low * (1 - w)
    """
    def __init__(self, channels, ratio=4):
        super().__init__()
        inter = channels // ratio

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_high, input_low):
        # Both inputs must have the same shape
        fused  = input_high + input_low
        w      = self.sigmoid(self.global_att(fused) + self.local_att(fused))
        return input_high * w + input_low * (1.0 - w)


# ── Decoder -- unchanged ──────────────────────────────────────────────────────
def _dw_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True),
    )

def decoder(inp, oup):
    return nn.Sequential(
        _dw_conv(inp, oup),
        nn.ConvTranspose2d(oup, oup, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True),
    )


# ── EMCAF-Net with AFF ────────────────────────────────────────────────────────
class nucleinet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = pvt_v2_b2()
        pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pvt_v2_b2.pth')
        if os.path.exists(pretrained_path):
            save_model = torch.load(pretrained_path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            matched = {k: v for k, v in save_model.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            self.backbone.load_state_dict(model_dict)
            print(f'[nucleinet] PVTv2-B2 weights: {len(matched)}/{len(model_dict)} backbone tensors matched')
            if len(matched) < 0.9 * len(model_dict):
                print('[nucleinet] WARNING: fewer than 90% of backbone weights matched.')
        else:
            print(f'[nucleinet] WARNING: pretrained weights not found at {pretrained_path}.')

        self.nsb4 = EMF(512); self.nsb3 = EMF(320)
        self.nsb2 = EMF(128); self.nsb1 = EMF(64)

        # CAF replaced with AFF
        self.afb4 = AFF(512); self.afb3 = AFF(320)
        self.afb2 = AFF(128); self.afb1 = AFF(64)

        self.dec4  = decoder(512, 320); self.dec3  = decoder(320, 128)
        self.dec2  = decoder(128, 64);  self.dec1  = decoder(64, 16)
        self.dec11 = decoder(16, 8)
        self.outc  = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        pvt = self.backbone(x)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]
        c4 = self.nsb4(x4); d4 = self.dec4(self.afb4(c4, x4))
        c3 = self.nsb3(x3); d3 = self.dec3(self.afb3(d4, c3))
        c2 = self.nsb2(x2); d2 = self.dec2(self.afb2(d3, c2))
        c1 = self.nsb1(x1); d1 = self.dec1(self.afb1(d2, c1))
        return self.outc(self.dec11(d1))


if __name__ == '__main__':
    from thop import profile
    model = nucleinet(); model.eval()
    dummy = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    N = 50; t0 = time.time()
    with torch.no_grad():
        for _ in range(N): model(dummy)
    fps = N / (time.time() - t0)
    print(f'[AFF replacing CAF]  Params: {params/1e6:.3f} M  |  GMACs: {flops/1e9:.3f}  |  FPS: {fps:.2f}')
