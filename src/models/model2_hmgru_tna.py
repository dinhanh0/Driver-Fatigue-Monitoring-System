from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

# --------- Temporal Neighborhood Attention (pure PyTorch, no custom CUDA) ----------
class TemporalNeighborhoodAttention(nn.Module):
    """
    Attention along the TIME axis only, in a local neighborhood (kernel_size).
    Input:  x (B, T, C)
    Output: y (B, T, C)
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, kernel_size: int = 9, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ks = kernel_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        v = self.v_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)

        # unfold neighborhood over time using tensor.unfold
        # to shape: k_unf, v_unf => (B,H,T,K,D)
        Ksz = self.ks
        pad = Ksz // 2  # symmetric
        # pad along T by replicating boundary values
        k_t = k.transpose(2, 3)                           # (B,H,D,T)
        k_t = F.pad(k_t, (pad, pad, 0, 0), mode='replicate')    # pad T dim (pad last dim only)
        k_unf = k_t.unfold(dimension=3, size=Ksz, step=1) # (B,H,D,T,K)
        k_unf = k_unf.permute(0,1,3,4,2)                  # (B,H,T,K,D)

        v_t = v.transpose(2, 3)                           # (B,H,D,T)
        v_t = F.pad(v_t, (pad, pad, 0, 0), mode='replicate')
        v_unf = v_t.unfold(dimension=3, size=Ksz, step=1) # (B,H,D,T,K)
        v_unf = v_unf.permute(0,1,3,4,2)                  # (B,H,T,K,D)

        # q ⋅ k^T over D => (B,H,T,K)
        attn_scores = (q.unsqueeze(3) * k_unf).sum(dim=-1) / (D ** 0.5)
        attn = F.softmax(attn_scores, dim=3)
        attn = self.attn_drop(attn)

        # weight the neighborhood values and sum over K => (B,H,T,D)
        y = (attn.unsqueeze(-1) * v_unf).sum(dim=3)       # (B,H,T,D)
        y = y.permute(0,2,1,3).reshape(B, T, C)           # (B,T,C)
        y = self.out_proj(y)
        return y

# --------- Gated Fusion ----------
class GatedFusion(nn.Module):
    """
    Fuse two sequences (B,T,C) with a learned gate:
      gate = sigmoid(Wv*V + Wf*F + b), out = gate*V + (1-gate)*F
    """
    def __init__(self, c_v: int, c_f: int, c_out: int):
        super().__init__()
        self.v_proj = nn.Linear(c_v, c_out)
        self.f_proj = nn.Linear(c_f, c_out)
        self.gate = nn.Linear(2*c_out, c_out)

    def forward(self, v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        # v, f: (B,T,channels)
        v_ = self.v_proj(v)
        f_ = self.f_proj(f)
        g = torch.sigmoid(self.gate(torch.cat([v_, f_], dim=-1)))
        return g * v_ + (1.0 - g) * f_

# --------- Model 2: HM-GRU + TNA + Gated Fusion ----------
class HMGRU_TNA(nn.Module):
    """
    Inputs:
      x_video: (B,T,3,224,224)  RGB frames
      x_feats: (B,T,Df)         handcrafted per-frame features (e.g., EAR/MAR/blur/illum) 
    Output:
      logits: (B,num_classes)
    """
    def __init__(
        self,
        num_classes: int = 2,
        feat_dim: int = 4,
        vis_embed_dim: int = 256,
        vis_gru_hidden: int = 128,
        feat_gru_hidden: int = 64,
        tna_heads: int = 4,
        tna_ks: int = 9,
        fuse_dim: int = 128,
        post_gru_hidden: int = 128,
        dropout: float = 0.2,
        imagenet_weights: bool = True,
    ):
        super().__init__()

        # 1) Visual encoder (frame-wise) using MobileNetV3-Small backbone
        if imagenet_weights:
            weights = "IMAGENET1K_V1"
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
            m = mobilenet_v3_small(weights=weights)
        else:
            m = mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(
            m.features,      # -> (B,576,H/32,W/32)
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),    # -> (B,576)
            nn.Linear(576, vis_embed_dim),
            nn.ReLU(inplace=True)
        )

        # 2) Visual temporal encoder: TNA + BiGRU
        self.tna = TemporalNeighborhoodAttention(vis_embed_dim, num_heads=tna_heads, kernel_size=tna_ks, dropout=dropout)
        self.vis_gru = nn.GRU(vis_embed_dim, vis_gru_hidden, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)

        # 3) Handcrafted feature stream: BiGRU
        self.feat_proj = nn.Linear(feat_dim, 32)
        self.feat_gru = nn.GRU(32, feat_gru_hidden, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)

        # 4) Gated Fusion
        self.fuse = GatedFusion(c_v=2*vis_gru_hidden, c_f=2*feat_gru_hidden, c_out=fuse_dim)

        # 5) Post-fusion temporal modeling + head
        self.post_gru = nn.GRU(fuse_dim, post_gru_hidden, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(2*post_gru_hidden * 2, 128),  # mean+max pooling concat → 2x
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_video: torch.Tensor, x_feats: torch.Tensor) -> torch.Tensor:
        """
        x_video: (B,T,3,224,224)
        x_feats: (B,T,Df)
        """
        B, T, C, H, W = x_video.shape
        # Encode each frame with CNN (flatten BT for speed)
        xv = x_video.view(B*T, C, H, W)
        with torch.amp.autocast('cuda', enabled=xv.is_cuda):  # ok on CPU too; AMP does nothing there
            emb = self.backbone(xv)                        # (B*T, vis_embed_dim)
        emb = emb.view(B, T, -1)                           # (B,T,Cv)

        # Temporal Neighborhood Attention on visual stream
        emb = self.tna(emb)                                # (B,T,Cv)

        # Visual BiGRU
        vis_out, _ = self.vis_gru(emb)                     # (B,T,2*vis_gru_hidden)

        # Handcrafted feature stream
        xf = self.feat_proj(x_feats)                       # (B,T,32)
        feat_out, _ = self.feat_gru(xf)                    # (B,T,2*feat_gru_hidden)

        # Gated Fusion
        fused = self.fuse(vis_out, feat_out)               # (B,T,fuse_dim)

        # Post-fusion BiGRU
        z, _ = self.post_gru(fused)                        # (B,T,2*post_gru_hidden)

        # Temporal pooling (mean + max)
        z_mean = z.mean(dim=1)
        z_max, _ = z.max(dim=1)
        z_cat = torch.cat([z_mean, z_max], dim=-1)

        # Head
        z_cat = self.dropout(z_cat)
        logits = self.cls(z_cat)                           # (B,num_classes)
        return logits