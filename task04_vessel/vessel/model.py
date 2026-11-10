"""Original compact residual U-Net, inspired by U-Net and spatial attention.

This is not a reproduction of SA-UNet, IterNet, or DSCNet.
"""
import torch
from torch import nn
from torch.nn import functional as F


def groups(channels):
    return next(n for n in [8, 4, 2, 1] if channels % n == 0)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups(cout), cout), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups(cout), cout),
        )
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        stats = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * torch.sigmoid(self.conv(stats))


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout, dropout):
        super().__init__()
        self.block = ResidualBlock(cin + skip, cout, dropout)
        self.attn = SpatialAttention()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.attn(self.block(torch.cat([x, skip], 1)))


class VesselUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=24, dropout=0.1):
        super().__init__()
        c = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList([ResidualBlock(in_channels, c[0])])
        self.encoder.extend(ResidualBlock(c[i-1], c[i], dropout) for i in range(1,5))
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList(DecoderBlock(c[i], c[i-1], c[i-1], dropout) for i in range(4,0,-1))
        self.head = nn.Conv2d(c[0], 1, 1)
        self.aux = nn.ModuleList([nn.Conv2d(c[1], 1, 1), nn.Conv2d(c[2], 1, 1)])

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='replicate')
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.pool(x))
            features.append(x)
        decoded = []
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = block(x, skip)
            decoded.append(x)
        logits = self.head(x)[..., :h, :w]
        aux = [head(feat)[..., : (h + scale - 1)//scale, : (w + scale - 1)//scale]
               for head, feat, scale in zip(self.aux, [decoded[-2], decoded[-3]], [2,4])]
        return {'logits': logits, 'aux': aux}


def build_model(config):
    return VesselUNet(config['in_channels'], config['base_channels'], config['dropout'])
