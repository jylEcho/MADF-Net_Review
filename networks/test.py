import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------- Swin related blocks -----------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

def window_partition(x, window_size):
    B, C, H, W = x.shape
    Ws_h, Ws_w = window_size
    x = x.view(B, C, H // Ws_h, Ws_h, W // Ws_w, Ws_w)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, Ws_h * Ws_w, C)
    return windows

def window_reverse(windows, window_size, H, W):
    Ws_h, Ws_w = window_size
    B = int(windows.shape[0] / (H // Ws_h * W // Ws_w))
    C = windows.shape[-1]
    x = windows.view(B, H // Ws_h, W // Ws_w, Ws_h, Ws_w, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.shift_size = (shift_size, shift_size) if isinstance(shift_size, int) else shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x).view(B, H, W, C)

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W

        shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) \
            if (self.shift_size[0] > 0 or self.shift_size[1] > 0) else x

        x_windows = shifted_x.permute(0, 3, 1, 2).contiguous()
        x_windows = window_partition(x_windows, self.window_size)

        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        shifted_x = shifted_x.permute(0, 2, 3, 1).contiguous()

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h != 0 or pad_w != 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H, W, C)
        x = self.mlp(self.norm2(x)) + x
        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + x

# ----------------- Simplified TransformerBlock (placeholder) -----------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.norm = nn.BatchNorm2d(dim)
    def forward(self, x):
        return x + self.norm(self.conv(x))

# ----------------- Mixed encoder builder -----------------
def make_mixed_blocks(dim, num_blocks, heads, input_resolution, LayerNorm_type, ffn_expansion_factor, bias):
    blocks = []
    for i in range(num_blocks):
        if i % 2 == 0:
            blocks.append(TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type))
        else:
            ws = 8
            shift = ws // 2
            blocks.append(SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=heads,
                                               window_size=ws, shift_size=shift))
    return nn.Sequential(*blocks)

# ----------------- Main Network -----------------
class HAformerSpatialFrequency(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=32, num_blocks=[2,2,2,2], heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.encoder_level1 = make_mixed_blocks(dim=int(dim*1), num_blocks=num_blocks[0], heads=heads[0],
                                               input_resolution=(256,256), LayerNorm_type=LayerNorm_type,
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.encoder_level2 = make_mixed_blocks(dim=int(dim*2), num_blocks=num_blocks[1], heads=heads[1],
                                               input_resolution=(128,128), LayerNorm_type=LayerNorm_type,
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.encoder_level3 = make_mixed_blocks(dim=int(dim*4), num_blocks=num_blocks[2], heads=heads[2],
                                               input_resolution=(64,64), LayerNorm_type=LayerNorm_type,
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.latent = make_mixed_blocks(dim=int(dim*8), num_blocks=num_blocks[3], heads=heads[3],
                                       input_resolution=(32,32), LayerNorm_type=LayerNorm_type,
                                       ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.output_proj = nn.Conv2d(int(dim*8), out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.encoder_level1(x)
        x2 = self.encoder_level2(F.avg_pool2d(x1, 2))
        x3 = self.encoder_level3(F.avg_pool2d(x2, 2))
        x4 = self.latent(F.avg_pool2d(x3, 2))
        out = self.output_proj(x4)
        return out

# ----------------- Test Run -----------------
if __name__ == "__main__":
    model = HAformerSpatialFrequency(dim=32)
    inp = torch.randn(1, 32, 256, 256)  # fake input with 32 channels
    out = model(inp)
    print("Input:", inp.shape)
    print("Output:", out.shape)
    print("Params (M):", sum(p.numel() for p in model.parameters())/1e6)
