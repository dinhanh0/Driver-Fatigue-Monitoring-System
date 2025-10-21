import torch
import torch.nn as nn
import torchvision.models as tv

# NAT (Neighborhood Attention) optional import
try:
    from natten import NeighborhoodAttention2D as NAT2D
    _HAS_NAT = True
except Exception:
    _HAS_NAT = False

class NATBlock(nn.Module):
    def __init__(self, dim=512, kernel_size=7, dilation=1):
        super().__init__()
        self.use_nat = _HAS_NAT
        if self.use_nat:
            self.attn = NAT2D(dim=dim, kernel_size=kernel_size, dilation=dilation)
        else:
            # Fallback: depthwise + pointwise conv stack as local-mixing approximation
            self.attn = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
            )
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, kernel_size=1),
        )
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = self.attn(x)
        x = self.norm1(x + h)
        h2 = self.ffn(x)
        x = self.norm2(x + h2)
        return x

class ResNet18LSTM(nn.Module):
    def __init__(self, num_classes=3, feat_dim=512, lstm_hidden=256, lstm_layers=1,
                 bidirectional=False, dropout=0.5, use_nat=False, nat_kernel=7, nat_dilation=1, nat_blocks=1,
                 backbone_name: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone_name
        self.feat_dim = feat_dim
        # Select backbone
        if backbone_name == "resnet18":
            m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
            backbone = nn.Sequential(*list(m.children())[:-2])  # -> (B,512,h,w)
            backbone_out_dim = 512
        elif backbone_name == "mobilenet_v2":
            m = tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V1)
            # mobilenet_v2.features outputs (B,1280,h,w)
            backbone = m.features
            backbone_out_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        # Adapter to project backbone channels to feat_dim if needed
        if backbone_out_dim != feat_dim:
            self.adapter = nn.Sequential(
                nn.Conv2d(backbone_out_dim, feat_dim, kernel_size=1),
                nn.BatchNorm2d(feat_dim),
                nn.GELU(),
            )
        else:
            self.adapter = nn.Identity()

        self.use_nat = use_nat
        if self.use_nat:
            self.nat = nn.Sequential(*[NATBlock(dim=feat_dim, kernel_size=nat_kernel, dilation=nat_dilation) for _ in range(nat_blocks)])
        else:
            self.nat = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout if lstm_layers>1 else 0.0)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def _encode_batch_frames(self, clip):
        # clip: (B,T,C,H,W) -> encode all frames in one backbone pass
        B, T, C, H, W = clip.shape
        x = clip.reshape(B * T, C, H, W)
        f = self.backbone(x)  # (B*T, backbone_ch, h, w)
        f = self.adapter(f)   # (B*T, feat_dim, h, w)
        f = self.nat(f)       # NAT refinement (or identity) on 4D
        f = self.pool(f).flatten(1)  # (B*T, feat_dim)
        seq = f.reshape(B, T, self.feat_dim)  # (B,T,feat_dim)
        return seq

    def forward(self, clip):     # clip: (B,T,C,H,W)
        # Encode all frames in a single, vectorized pass through the backbone
        seq = self._encode_batch_frames(clip)
        out, _ = self.lstm(seq)
        logits = self.head(out[:, -1])
        return logits
