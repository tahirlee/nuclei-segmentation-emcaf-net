"""
Dual-decoder U-Net (PyTorch re-implementation of the Keras model `dual_decoder_unet_binary`
of Mahbod et al., 2022, "A dual decoder U-Net-based model for nuclei instance segmentation in
hematoxylin and eosin-stained histological images", Front. Med.).

One shared encoder and two identical decoders:
    decoder 1 -> distance map (linear output, trained with MSE)
    decoder 2 -> binary segmentation (logit output; sigmoid is applied in the loss / at inference)

Layer sizes, dropout rates and he_normal initialisation follow the Keras code
(16-32-64-128-256/512 channels, 3x3 convolutions, 2x2 transposed convolutions).

The class is called `nucleinet` on purpose, so that train / test scripts that import
`model.<arch>.nucleinet` work unchanged:
    training mode:    forward(x) -> (segmentation_logit, distance_map)
    evaluation mode:  forward(x) -> segmentation_logit      (the distance decoder is not run)
"""
import torch
import torch.nn as nn


def _conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True))


class _Block(nn.Module):
    """conv-ReLU - dropout - conv-ReLU  (as in the Keras code)"""
    def __init__(self, i, o, p, o2=None):
        super().__init__()
        self.a, self.drop, self.b = _conv(i, o), nn.Dropout(p), _conv(o, o2 or o)

    def forward(self, x):
        return self.b(self.drop(self.a(x)))


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.u6, self.c6 = nn.ConvTranspose2d(512, 128, 2, 2), _Block(256, 128, 0.2)
        self.u7, self.c7 = nn.ConvTranspose2d(128, 64, 2, 2), _Block(128, 64, 0.2)
        self.u8, self.c8 = nn.ConvTranspose2d(64, 32, 2, 2), _Block(64, 32, 0.1)
        self.u9, self.c9 = nn.ConvTranspose2d(32, 16, 2, 2), _Block(32, 16, 0.1)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, c1, c2, c3, c4, c5):
        x = self.c6(torch.cat([self.u6(c5), c4], 1))
        x = self.c7(torch.cat([self.u7(x), c3], 1))
        x = self.c8(torch.cat([self.u8(x), c2], 1))
        x = self.c9(torch.cat([self.u9(x), c1], 1))
        return self.out(x)


class nucleinet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2 = _Block(3, 16, 0.1), _Block(16, 32, 0.1)
        self.e3, self.e4 = _Block(32, 64, 0.2), _Block(64, 128, 0.2)
        self.e5 = _Block(128, 256, 0.1, o2=512)
        self.pool = nn.MaxPool2d(2)
        self.dec_dist = _Decoder()      # distance-map decoder
        self.dec_seg = _Decoder()       # segmentation decoder
        for m in self.modules():        # Keras he_normal
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        c4 = self.e4(self.pool(c3))
        c5 = self.e5(self.pool(c4))
        seg = self.dec_seg(c1, c2, c3, c4, c5)
        if self.training:
            return seg, self.dec_dist(c1, c2, c3, c4, c5)
        return seg


if __name__ == '__main__':
    m = nucleinet()
    n = sum(p.numel() for p in m.parameters())
    n_seg = n - sum(p.numel() for p in m.dec_dist.parameters())
    print(f'Params (both decoders, training): {n / 1e6:.3f} M   |   inference (segmentation decoder only): {n_seg / 1e6:.3f} M')
    x = torch.randn(1, 3, 512, 512)
    m.train(); s, d = m(x); print('train:', tuple(s.shape), tuple(d.shape))
    m.eval(); print('eval :', tuple(m(x).shape))
