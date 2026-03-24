import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g1 = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionZoneNet(nn.Module):
    """
    Saliency-like head:
    observed image -> saliency map in [0,1]
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        saliency = self.sigmoid(self.head(d2))
        return saliency


def saliency_to_soft_zones(saliency: torch.Tensor, low: float, high: float, temp: float):
    """
    saliency: [B,1,H,W]
    output zone_probs: [B,3,H,W]
    zone order: 0=attention, 1=intermediate, 2=around
    """
    attn = torch.sigmoid((saliency - high) / temp)
    around = torch.sigmoid((low - saliency) / temp)
    inter = 1.0 - attn - around
    inter = inter.clamp(min=1e-6)

    zone_probs = torch.cat([attn, inter, around], dim=1)
    zone_probs = zone_probs / (zone_probs.sum(dim=1, keepdim=True) + 1e-8)
    return zone_probs


def saliency_to_hard_zones(saliency: torch.Tensor, low: float, high: float):
    """
    output zone_map: [B,H,W], int64
    0=attention, 1=intermediate, 2=around
    """
    z = torch.ones_like(saliency, dtype=torch.long)  # intermediate
    z = torch.where(saliency >= high, torch.zeros_like(z), z)
    z = torch.where(saliency < low, torch.full_like(z, 2), z)
    return z.squeeze(1)