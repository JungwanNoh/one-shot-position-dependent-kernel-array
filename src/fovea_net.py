import torch
import torch.nn as nn
from config import CFG


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FoveaParamNet(nn.Module):
    """
    Predict per-image foveated parameters:
    cx, cy, r1, r2
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)

        raw = self.fc(x)  # [B,4]

        cx = torch.sigmoid(raw[:, 0])
        cy = torch.sigmoid(raw[:, 1])

        r1 = CFG.R1_MIN + (CFG.R1_MAX - CFG.R1_MIN) * torch.sigmoid(raw[:, 2])
        dr = CFG.DR_MIN + (CFG.DR_MAX - CFG.DR_MIN) * torch.sigmoid(raw[:, 3])
        r2 = r1 + dr

        return {
            "cx": cx,
            "cy": cy,
            "r1": r1,
            "r2": r2,
        }


def build_distance_map(h: int, w: int, cx: torch.Tensor, cy: torch.Tensor, device: str):
    """
    cx, cy: [B]
    output: [B,1,H,W]
    """
    yy = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, w)

    yy = yy.expand(cx.shape[0], h, w)
    xx = xx.expand(cx.shape[0], h, w)

    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)

    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-8)
    return dist.unsqueeze(1)  # [B,1,H,W]


def params_to_soft_zones(params: dict, h: int, w: int, device: str):
    """
    zone order: 0=attention, 1=intermediate, 2=around
    """
    dist = build_distance_map(h, w, params["cx"], params["cy"], device)

    r1 = params["r1"].view(-1, 1, 1, 1)
    r2 = params["r2"].view(-1, 1, 1, 1)

    attn = torch.sigmoid((r1 - dist) / CFG.SOFT_TEMP)
    around = torch.sigmoid((dist - r2) / CFG.SOFT_TEMP)
    inter = (1.0 - attn - around).clamp(min=1e-6)

    zone_probs = torch.cat([attn, inter, around], dim=1)
    zone_probs = zone_probs / (zone_probs.sum(dim=1, keepdim=True) + 1e-8)
    return zone_probs, dist


def params_to_hard_zones(params: dict, h: int, w: int, device: str):
    dist = build_distance_map(h, w, params["cx"], params["cy"], device)

    r1 = params["r1"].view(-1, 1, 1, 1)
    r2 = params["r2"].view(-1, 1, 1, 1)

    zone_map = torch.full_like(dist, 2, dtype=torch.long)   # around
    zone_map = torch.where(dist <= r2, torch.ones_like(zone_map), zone_map)  # intermediate
    zone_map = torch.where(dist <= r1, torch.zeros_like(zone_map), zone_map) # attention

    return zone_map.squeeze(1), dist