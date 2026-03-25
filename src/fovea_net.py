import torch
import torch.nn as nn
import torch.nn.functional as F

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


def softmax_heatmap_to_center(logits: torch.Tensor, temp: float):
    """
    logits: [B,1,H,W]
    return:
      heatmap_probs: [B,1,H,W]
      cx, cy: [B]
    """
    b, _, h, w = logits.shape
    flat = logits.view(b, -1) / temp
    probs = F.softmax(flat, dim=1).view(b, 1, h, w)

    yy = torch.linspace(0.0, 1.0, h, device=logits.device).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=logits.device).view(1, 1, 1, w)

    cx = (probs * xx).sum(dim=(1, 2, 3))
    cy = (probs * yy).sum(dim=(1, 2, 3))

    return probs, cx, cy


class FoveaHeatmapNet(nn.Module):
    """
    Predict:
      - spatial heatmap for center
      - scalar radii r1, r2 from bottleneck
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch)           # H, W
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)     # H/2, W/2
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # H/4, W/4

        # decoder for heatmap
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        self.heatmap_head = nn.Conv2d(base_ch, 1, kernel_size=1)

        # radius head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.radius_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

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

        heatmap_logits = self.heatmap_head(d2)
        heatmap_probs, cx, cy = softmax_heatmap_to_center(
            heatmap_logits,
            temp=CFG.HEATMAP_SOFTMAX_TEMP,
        )

        raw_radius = self.radius_head(self.gap(b))
        r1 = CFG.R1_MIN + (CFG.R1_MAX - CFG.R1_MIN) * torch.sigmoid(raw_radius[:, 0])
        dr = CFG.DR_MIN + (CFG.DR_MAX - CFG.DR_MIN) * torch.sigmoid(raw_radius[:, 1])
        r2 = r1 + dr

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap_probs": heatmap_probs,
            "cx": cx,
            "cy": cy,
            "r1": r1,
            "r2": r2,
        }


def build_distance_map(h: int, w: int, cx: torch.Tensor, cy: torch.Tensor, device: str):
    yy = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, w)

    yy = yy.expand(cx.shape[0], h, w)
    xx = xx.expand(cx.shape[0], h, w)

    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)

    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-8)
    return dist.unsqueeze(1)  # [B,1,H,W]


def params_to_soft_zones(params: dict, h: int, w: int, device: str):
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