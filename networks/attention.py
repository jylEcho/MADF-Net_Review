import torch
import torch.nn as nn
import math


class MultiSinglePeriodCrossAttention(nn.Module):
    """
    三个单期与融合通道的跨注意力（线性注意力版本）
    输入:
        t1, t2, t3: [B, C, H, W]  # 单期特征
        fusion_tensor: [B, C, H, W]  # 融合特征
    输出:
        fused_out: [B, C, H, W]
    """
    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # 初始化投影层
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.back_proj = None
        self.out_proj = None
        self.channels = None

        self.dropout = nn.Dropout(dropout)

    def _init_projections(self, channels, device):
        """显式初始化投影层"""
        if self.channels != channels or self.q_proj is None:
            self.q_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.k_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.v_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.back_proj = nn.Conv2d(self.d_model, channels, kernel_size=1, bias=False).to(device)
            self.out_proj = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False).to(device)
            self.channels = channels

    def _to_seq(self, x):
        B, C, H, W = x.shape
        N = H * W
        return x.view(B, C, N).permute(0, 2, 1), H, W  # [B, N, C], H, W

    def _from_seq(self, x_seq, H, W):
        B, N, C = x_seq.shape
        return x_seq.permute(0, 2, 1).contiguous().view(B, C, H, W)

    def cross_attn(self, single_feat, fusion_feat):
        """
        线性注意力版本，避免 N×N 显存消耗
        """
        q = self.q_proj(single_feat)
        k = self.k_proj(fusion_feat)
        v = self.v_proj(fusion_feat)

        q_seq, H, W = self._to_seq(q)  # [B, N, D]
        k_seq, _, _ = self._to_seq(k)  # [B, N, D]
        v_seq, _, _ = self._to_seq(v)  # [B, N, D]

        B, N, D = q_seq.shape
        Nh, Hd = self.num_heads, self.head_dim

        qh = q_seq.view(B, N, Nh, Hd)  # [B, N, Nh, Hd]
        kh = k_seq.view(B, N, Nh, Hd)
        vh = v_seq.view(B, N, Nh, Hd)

        # ---- Linear Attention Trick ----
        qh = torch.relu(qh) + 1e-6  # 保证正值
        kh = torch.relu(kh) + 1e-6

        # KᵀV 部分 (B, Nh, Hd, Hd)
        kv = torch.einsum("bnhd,bnhm->bhdm", kh, vh)

        # Normalization: 计算分母 Z
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", qh, kh.sum(dim=1)) + 1e-6)

        # 输出 (B, N, Nh, Hd)
        out = torch.einsum("bnhd,bhdm,bnh->bnhm", qh, kv, z)

        out = out.reshape(B, N, D)  # [B, N, D]
        out_spatial = self._from_seq(out, H, W)
        return self.back_proj(out_spatial)

    def forward(self, t1, t2, t3, fusion_tensor):
        device = t1.device
        self._init_projections(t1.shape[1], device)

        out1 = self.cross_attn(t1, fusion_tensor)
        out2 = self.cross_attn(t2, fusion_tensor)
        out3 = self.cross_attn(t3, fusion_tensor)

        concat_out = torch.cat([out1, out2, out3], dim=1)
        fused_out = self.out_proj(concat_out)
        return fused_out
