"""
Ablation Variant 6: EMCAF-Net with EMCAD-style block replacing EMF.
Reference: Rahman et al., "EMCAD: Efficient Multi-Scale Convolutional Attention
Decoding for Medical Image Segmentation", CVPR 2024.

The EMCAD-style block uses:
  - Multi-scale depthwise convolutions (3x3, 5x5, 7x7) concatenated
  - Pointwise (1x1) projection back to in_channel
  - GAP-based channel attention gate
  - Residual skip connection
It does NOT use edge-subtraction, a gated PReLU enhancement, or dilated convolutions.
CAF/AFB block is unchanged.
"""
import os, sys, time
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.pvtv2 import pvt_v2_b2


# ── EMCAD-style multi-scale block (replaces EMF) ──────────────────────────────
class EMCADBlock(nn.Module):
    """
    Efficient Multi-Scale Convolutional Attention block (EMCAD-style).
    Replaces EMF to compare it with an existing multi-scale attention block.

    Architecture:
      Input x (C, H, W)
        |-- DWConv 3x3  --|
        |-- DWConv 5x5  --|-> Cat (3C, H, W) -> PW Conv (3C->C) -> BN -> ReLU -> ms
        |-- DWConv 7x7  --|
      Channel attention: GAP(ms) -> FC(C->C//r) -> ReLU -> FC(C//r->C) -> Sigmoid -> w
      Output: ms * w + x   (channel attention + residual)
    """
    def __init__(self, in_channel, ratio=4):
        super().__init__()
        # Multi-scale depthwise convolutions
        self.dw3 = nn.Conv2d(in_channel, in_channel, 3, padding=1,  groups=in_channel, bias=False)
        self.dw5 = nn.Conv2d(in_channel, in_channel, 5, padding=2,  groups=in_channel, bias=False)
        self.dw7 = nn.Conv2d(in_channel, in_channel, 7, padding=3,  groups=in_channel, bias=False)
        # Pointwise projection
        self.pw  = nn.Conv2d(in_channel * 3, in_channel, 1, bias=False)
        self.bn  = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        # Channel attention gate
        inter = max(in_channel // ratio, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, inter, 1, bias=False)
        self.fc2 = nn.Conv2d(inter, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Multi-scale feature extraction
        ms = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        ms = self.relu(self.bn(self.pw(ms)))
        # Channel attention
        w  = self.sigmoid(self.fc2(self.relu(self.fc1(self.gap(ms)))))
        return ms * w + x    # residual


# ── CAF/AFB -- unchanged ──────────────────────────────────────────────────────
class AFB(nn.Module):
    def __init__(self, mid_ch, out_ch):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_high = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.Sigmoid())
        self.branch_low = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.Sigmoid())

    def forward(self, input_high, input_low):
        w_high = self.branch_high(self.global_pool(input_high))
        w_low  = self.branch_low( self.global_pool(input_low))
        return input_high * w_high + input_low * w_low


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


# ── EMCAF-Net with EMCAD-style block ─────────────────────────────────────────
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

        # EMF replaced with EMCAD-style block
        self.nsb4 = EMCADBlock(512); self.nsb3 = EMCADBlock(320)
        self.nsb2 = EMCADBlock(128); self.nsb1 = EMCADBlock(64)

        self.afb4 = AFB(512, 512); self.afb3 = AFB(320, 320)
        self.afb2 = AFB(128, 128); self.afb1 = AFB(64, 64)
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
    print(f'[EMCAD-style (EMF replaced)]  Params: {params/1e6:.3f} M  |  GMACs: {flops/1e9:.3f}  |  FPS: {fps:.2f}')
