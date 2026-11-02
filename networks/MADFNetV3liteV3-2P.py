import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up


# =========================
# Basic light modules
# =========================
class DSConv(nn.Module):
    """Depthwise separable conv: much cheaper than normal 3x3 conv."""
    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LiteUNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            DSConv(in_channels, out_channels, stride=stride, act=True),
            DSConv(out_channels, out_channels, stride=1, act=True),
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(4, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class FouierConv(nn.Module):
    """Original frequency branch, kept but lighter SE reduction."""
    def __init__(self, n_feat):
        super().__init__()
        self.main = nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat)
        self.mag = nn.Sequential(
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 1, 1, 0, groups=n_feat),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
        )
        self.pha = nn.Sequential(
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 1, 1, 0, groups=n_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 1, 1, 0, groups=n_feat),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
        )
        self.pha_se = SEBlock(n_feat)
        self.mag_se = SEBlock(n_feat)

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)

        mag_out = self.mag_se(self.mag(mag))
        pha_out = self.pha_se(self.pha(pha))

        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return y + self.main(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            1,
            1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Removed unused qkv/project_out from the old Attention to reduce params."""
    def __init__(self, dim, num_heads, bias, use_fourier=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv2 = nn.Conv2d(dim // 2, dim * 3 // 2, 1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(
            dim * 3 // 2,
            dim * 3 // 2,
            3,
            1,
            1,
            groups=dim * 3 // 2,
            bias=bias,
        )
        self.project_out2 = nn.Conv2d(dim // 2, dim // 2, 1, bias=bias)
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.use_fourier = use_fourier
        self.fourier = FouierConv(dim // 2) if use_fourier else None

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = self.fourier(x1) if self.use_fourier else x1

        b, c, h, w = x2.shape
        qkv = self.qkv_dwconv2(self.qkv2(x2))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.avgpool2(k)
        v = self.avgpool2(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (k.transpose(-2, -1) @ q)
        attn = attn.softmax(dim=-2)
        out2 = (v @ attn)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)
        return torch.cat([out, out2], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_fourier=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, use_fourier=use_fourier)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm3(x))
        return x


class LiteFusion(nn.Module):
    """
    Replace INNFusion/RevFusionBlock.
    Much cheaper: concat -> 1x1 compress -> depthwise local refine -> SE gate.
    """
    def __init__(self, channels):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.refine = DSConv(channels, channels, stride=1, act=True)
        self.gate = SEBlock(channels, reduction=8)

    def forward(self, x1, x2):
        x = self.compress(torch.cat([x1, x2], dim=1))
        x = self.refine(x)
        return self.gate(x) + x1


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    """Original normal stride-2 conv -> depthwise separable stride-2 conv."""
    def __init__(self, n_feat):
        super().__init__()
        self.down_sample = DSConv(n_feat, n_feat * 2, stride=2, act=True)

    def forward(self, x):
        return self.down_sample(x)


class BranchAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_list):
        b, c, h, w = feat_list[0].shape
        scores = []

        for x in feat_list:
            v = F.adaptive_avg_pool2d(x, 1).view(b, c)
            scores.append(self.mlp(v))

        scores = torch.stack(scores, dim=1)  # (b, n, 1)
        weights = torch.softmax(scores, dim=1)

        fused = 0
        for i in range(len(feat_list)):
            fused = fused + weights[:, i].view(b, 1, 1, 1) * feat_list[i]

        return fused, weights

class HAformerSpatialFrequency(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=40,
        num_blocks=[0, 1, 1, 2],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=1.5,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
    ):
        super().__init__()

        # Main branch: concat(art, pv)
        self.stem_art_pv = OverlapPatchEmbed(1, dim, bias=bias)

        # Auxiliary branches only keep three independent stems + level1 processing.
        self.stem_art = OverlapPatchEmbed(1, dim, bias=bias)
        self.stem_pv = OverlapPatchEmbed(1, dim, bias=bias)
        self.stem_delay = OverlapPatchEmbed(1, dim, bias=bias)

        self.branch_att_level1 = BranchAttention(dim)

        # Lightweight fusion modules.
        self.fusion_module1 = LiteFusion(dim)
        self.fusion_module2 = LiteFusion(dim * 2)
        self.fusion_module3 = LiteFusion(dim * 4)
        self.fusion_module4 = LiteFusion(dim * 8)

        # Level1 keeps ART/PV/Delay separation, which is most useful for modality-specific low-level information.
        self.conv_block_art1 = LiteUNetConvBlock(dim, dim, stride=1)
        self.conv_block_pv1 = LiteUNetConvBlock(dim, dim, stride=1)
        self.conv_block_delay1 = LiteUNetConvBlock(dim, dim, stride=1)

        # Level2-4 remove repeated 3-branch computation. After level1, aux is already fused.
        self.conv_block_aux2 = LiteUNetConvBlock(dim * 2, dim * 2, stride=1)
        self.conv_block_aux3 = LiteUNetConvBlock(dim * 4, dim * 4, stride=1)
        self.conv_block_aux4 = LiteUNetConvBlock(dim * 8, dim * 8, stride=1)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                use_fourier=False,  # high-res fourier is expensive; level1 has 0 block by default anyway
            ) for _ in range(num_blocks[0])
        ])

        self.down_ap_1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 2,
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                use_fourier=True,
            ) for _ in range(num_blocks[1])
        ])

        self.down_ap_2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 4,
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                use_fourier=True,
            ) for _ in range(num_blocks[2])
        ])

        self.down_ap_3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 8,
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                use_fourier=True,
            ) for _ in range(num_blocks[3])
        ])

        self.downsample_aux1 = LiteUNetConvBlock(dim, dim * 2, stride=2)
        self.downsample_aux2 = LiteUNetConvBlock(dim * 2, dim * 4, stride=2)
        self.downsample_aux3 = LiteUNetConvBlock(dim * 4, dim * 8, stride=2)

        # Dynamic channels. No more hard-coded 128/256/512/64.
        self.up1 = Up(in_channels=dim * 2, out_channels=dim, layers=1)
        self.up2 = Up(in_channels=dim * 4, out_channels=dim, layers=2)
        self.up3 = Up(in_channels=dim * 8, out_channels=dim, layers=3)
        self.convout = nn.Conv2d(dim * 4, out_channels, 3, 1, 1, bias=bias)

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)

        # Kept for compatibility with old checkpoints/scripts, not used in forward.
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)
        self.output_artpv = nn.Conv2d(2, out_channels, 1, bias=bias)

    def forward(self, art, delay):
        # 主分支（只用ART）
        feat_main = self.stem_art_pv(art)
    
        # 两个辅助分支
        feat_art = self.stem_art(art)
        feat_dl = self.stem_delay(delay)
    
        # Stage 1
        out_enc_level1_main = self.encoder_level1(feat_main)
    
        out_art_l1 = self.conv_block_art1(feat_art)
        out_dl_l1 = self.conv_block_delay1(feat_dl)
    
        fused_aux_l1, _ = self.branch_att_level1([out_art_l1, out_dl_l1])
    
        fused_level1 = self.fusion_module1(out_enc_level1_main, fused_aux_l1)
    
        art_down1 = self.down_ap_1_2(fused_level1 + out_enc_level1_main)
        aux_down1 = self.downsample_aux1(fused_level1 + fused_aux_l1)
    
        # Stage 2
        out_enc_level2_main = self.encoder_level2(art_down1)
        fused_aux_l2 = self.conv_block_aux2(aux_down1)
        fused_level2 = self.fusion_module2(out_enc_level2_main, fused_aux_l2)
    
        art_down2 = self.down_ap_2_3(fused_level2 + out_enc_level2_main)
        aux_down2 = self.downsample_aux2(fused_level2 + fused_aux_l2)
    
        # Stage 3
        out_enc_level3_main = self.encoder_level3(art_down2)
        fused_aux_l3 = self.conv_block_aux3(aux_down2)
        fused_level3 = self.fusion_module3(out_enc_level3_main, fused_aux_l3)
    
        art_down3 = self.down_ap_3_4(fused_level3 + out_enc_level3_main)
        aux_down3 = self.downsample_aux3(fused_level3 + fused_aux_l3)
    
        # Stage 4
        out_enc_level4_main = self.latent(art_down3)
        fused_aux_l4 = self.conv_block_aux4(aux_down3)
        fusion_level4 = self.fusion_module4(out_enc_level4_main, fused_aux_l4)
    
        # Upsample
        up1 = self.up1(fused_level2)
        up2 = self.up2(fused_level3)
        up3 = self.up3(fusion_level4)
    
        output = torch.cat([up1, up2, up3, fused_level1], dim=1)
        output = self.convout(output)
    
        return output

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)


if __name__ == "__main__":
    from thop import profile

    model = HAformerSpatialFrequency(
        dim=40,
        num_blocks=[0, 1, 1, 2],
        ffn_expansion_factor=1.5,
    ).cuda('cuda:0')
    model.apply(weights_init)

    x = torch.rand((1, 1, 224, 224)).cuda('cuda:0')
    flops, params = profile(model, inputs=(x, x[:]))

    print(f"FLOPs: {flops / 1e9} G FLOPs")
    print(f"模型参数数量: {params / 1e6} M")
