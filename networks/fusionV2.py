# HAformer Spatial-Frequency model (Attention shape-bug fix)
# This file is identical to the modified version but fixes the Attention forward shapes bug reported.

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch.nn.init as init
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pdb import set_trace as stx
# import numbers

# from einops import rearrange
# from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
# from MSDNN import InvBlock,HinResBlock
import sys
import os
# 将当前文件所在目录添加到Python搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
from MSDNN import InvBlock,HinResBlock

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        return x * y


class FouierConv(nn.Module):
    def __init__(self, n_feat):
        super(FouierConv, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=n_feat)
        self.mag = nn.Sequential(nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
                                 nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(inplace=True),
                                 )
        self.pha = nn.Sequential(nn.BatchNorm2d(n_feat),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=n_feat),
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
        norm_mag = mag
        norm_pha = pha
        mag_out = self.mag_se(self.mag(norm_mag))
        pha_out = self.pha_se(self.pha(norm_pha))
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
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-Scale Attention (fixed)
class Attention(nn.Module):
    """
    MultiScaleAttention
    - Keeps Fourier branch for one half.
    - Multi-scale pooled keys/values for the other half.
    - Correct q/k/v shapes to avoid mismatched matmuls.
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.fourier = FouierConv(dim // 2)
        self.q_proj = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim // 2, dim // 2 * 2, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)
        self.pool_merger = nn.Sequential(
            nn.Conv2d(dim // 2 * 3, dim // 2, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(8)
        self.pool3 = nn.AdaptiveAvgPool2d(16)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out_fourier = self.fourier(x1)
        b, c, h, w = x2.shape

        # queries -> [b, heads, HW, head_dim]
        q = self.q_proj(x2)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        # multi-scale pooled keys/values
        k1 = self.pool1(x2)
        k2 = self.pool2(x2)
        k3 = self.pool3(x2)
        k1_r = F.adaptive_avg_pool2d(k1, 8)
        k2_r = k2
        k3_r = F.adaptive_avg_pool2d(k3, 8)
        merged = torch.cat([k1_r, k2_r, k3_r], dim=1)
        merged = self.pool_merger(merged)

        kv = self.kv_proj(merged)
        k, v = kv.chunk(2, dim=1)

        # reshape k,v -> [b, heads, HkWk, head_dim]
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_att = attn @ v
        out = rearrange(out_att, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.out_proj(out)
        return torch.cat([out_fourier, out], dim=1)


##########################################################################
# Cross-gated fusion (replacement for INNFusion)
class CrossGatedFusion(nn.Module):
    def __init__(self, channels):
        super(CrossGatedFusion, self).__init__()
        self.channels = channels
        self.fuse_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.gate_x1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gate_x2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )
        self.se = SEBlock(channels)

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        base = self.fuse_conv(concat)
        g1 = self.gate_x1(concat)
        g2 = self.gate_x2(concat)
        gated = base * g1 + base * g2
        fused = gated + self.refine(gated)
        fused = self.se(fused)
        return fused


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_feat*2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.down_sample(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(n_feat//2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up_sample(x)


##########################################################################
##---------- Restormer-like HAformer -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm3(x))
        return x


######################## fusion model ##################################
class HAformerSpatialFrequency(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 3, 3, 4],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):

        super(HAformerSpatialFrequency, self).__init__()

        self.fusion_module1 = CrossGatedFusion(dim)
        self.fusion_module2 = CrossGatedFusion(dim * 2)
        self.fusion_module3 = CrossGatedFusion(dim * 4)
        self.fusion_module4 = CrossGatedFusion(dim * 8)

        self.conv_block_delay1 = UNetConvBlock(dim, dim, stride=1)
        self.conv_block_delay2 = UNetConvBlock(dim*2, dim*2, stride=1)
        self.conv_block_delay3 = UNetConvBlock(dim*4, dim*4, stride=1)
        self.conv_block_delay4 = UNetConvBlock(dim * 8, dim * 8, stride=1)

        self.downsample_delay1 = UNetConvBlock(dim, dim * 2)
        self.downsample_delay2 = UNetConvBlock(dim * 2, dim * 4)
        self.downsample_delay3 = UNetConvBlock(dim * 4, dim * 8)

        self.stem_art_pv = OverlapPatchEmbed(2, dim)
        self.stem_delay = OverlapPatchEmbed(1, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_ap_1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down_ap_2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down_ap_3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up1 = Up(in_channels=128, out_channels=64, layers=1)
        self.up2 = Up(in_channels=256, out_channels=64, layers=2)
        self.up3 = Up(in_channels=512, out_channels=64, layers=3)

        self.convout = nn.Conv2d(256,3,kernel_size=3, stride=1, padding=1, bias=bias)

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_artpv = nn.Conv2d(2, out_channels, kernel_size=1, bias=bias)

    def forward(self, art, pv, delay):
        art_pv = torch.cat([art, pv], dim=1)
        feat_art_pv = self.stem_art_pv(art_pv)
        feat_delay = self.stem_delay(delay)

        out_enc_level1_delay = self.conv_block_delay1(feat_delay)
        out_enc_level1_art_pv = self.encoder_level1(feat_art_pv)
        fused_level1 = self.fusion_module1(out_enc_level1_art_pv, out_enc_level1_delay)

        art_pv_downsampled1 = self.down_ap_1_2(fused_level1 + out_enc_level1_art_pv)
        delay_downsampled1 = self.downsample_delay1(fused_level1 + out_enc_level1_delay)

        out_enc_level2_delay = self.conv_block_delay2(delay_downsampled1)
        out_enc_level2_art_pv = self.encoder_level2(art_pv_downsampled1)
        fused_level2 = self.fusion_module2(out_enc_level2_art_pv, out_enc_level2_delay)

        art_pv_downsampled2 = self.down_ap_2_3(fused_level2+out_enc_level2_art_pv)
        delay_downsampled2 = self.downsample_delay2(fused_level2 + out_enc_level2_delay)

        out_enc_level3_delay = self.conv_block_delay3(delay_downsampled2)
        out_enc_level3_art_pv = self.encoder_level3(art_pv_downsampled2)
        fused_level3 = self.fusion_module3(out_enc_level3_art_pv, out_enc_level3_delay)

        art_pv_downsampled3 = self.down_ap_3_4(fused_level3 + out_enc_level3_art_pv)
        delay_downsampled3 = self.downsample_delay3(fused_level3 + out_enc_level3_delay)

        out_enc_level4_delay = self.conv_block_delay4(delay_downsampled3)
        out_enc_level4_art_pv = self.latent(art_pv_downsampled3)
        fusion_level4 = self.fusion_module4(out_enc_level4_art_pv, out_enc_level4_delay)

        up1 = self.up1(fused_level2)
        up2 = self.up2(fused_level3)
        up3 = self.up3(fusion_level4)
        output = torch.cat([up1, up2, up3, fused_level1], 1)
        output = self.convout(output)

        return output


def weights_init(m):
    if isinstance(m, nn.Conv2d):
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

    model = HAformerSpatialFrequency().cuda('cuda:0')
    model.apply(weights_init)
    x = torch.rand((1, 1, 224, 224)).cuda('cuda:0')
    # z = torch.randn(1,64,64,65)
    # se = SEBlock(64)
    # print(se(z))
    # 使用 thop 计算 FLOPs
    flops, params = profile(model, inputs=(x,x[:],x[:]))

    print(f"FLOPs: {flops / 1e9} G FLOPs")  # 以十亿（Giga）FLOPs为单位打印
    print(f"模型参数数量: {params / 1e6} M")
    # import ipdb;ipdb.set_trace()
# if __name__ == '__main__':
#     model = HAformerSpatialFrequency().cuda('cuda:0')
#     model.apply(weights_init)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#     if torch.cuda.is_available():
#         print(f"GPU型号: {torch.cuda.get_device_name(0)}")

#     print("\n===== 模型结构概览 =====")
#     print(model)

#     print("\n===== 输入验证 =====")
#     batch_size = 6
#     img_size = 224

#     image_batch_a = torch.rand(batch_size, 1, img_size, img_size).to(device)
#     image_batch_v = torch.rand(batch_size, 1, img_size, img_size).to(device)
#     image_batch_d = torch.rand(batch_size, 1, img_size, img_size).to(device)

#     print(f"输入A尺寸: {image_batch_a.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")
#     print(f"输入V尺寸: {image_batch_v.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")
#     print(f"输入D尺寸: {image_batch_d.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")

#     print("\n===== 前向传播测试（评估模式） =====")
#     model.eval()
#     with torch.no_grad():
#         output = model(image_batch_a, image_batch_v, image_batch_d)

#     print(f"输出尺寸: {output.shape} (预期: [{batch_size}, 3, {img_size}, {img_size}])")
