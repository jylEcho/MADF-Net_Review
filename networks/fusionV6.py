# Merged: fusion_version5 base + minimal integration of ideas from V1.0
# - Kept the original HAformerSpatialFrequency architecture from fusion_version5.py
# - Added a small, self-contained FeatureConcatAndRestore2D class adapted from V1.0's
#   FeatureConcatAndRestore idea to perform multi-feature concatenation + projection
# - Replaced the final naive torch.cat([up1,up2,up3,fused_level1],1) with a call to
#   FeatureConcatAndRestore2D to minimally "inject" the author's fusion idea.
# The change is intentionally small and localized: only a new helper class and a single
# line change in forward() are added. This preserves the original model behavior/shape
# while introducing the user's fusion idea.

import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from .up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
from .MSDNN import InvBlock,HinResBlock

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

        # Global average pooling
        y = self.pool(x).view(batch_size, channels)

        # Squeeze
        y = self.fc1(y)
        y = self.relu(y)

        # Excitation
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

        # norm_mag = (mag - torch.mean(mag)) / torch.std(mag)
        # norm_pha = (pha - torch.mean(pha)) / torch.std(pha)
        norm_mag = mag
        norm_pha = pha
        if torch.isnan(norm_mag.any()):
            print('nan')

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
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim // 2, dim * 3 // 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, kernel_size=3, stride=1, padding=1, groups=dim * 3 // 2,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)

        self.qkv2 = nn.Conv2d(dim // 2, dim * 3 // 2, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, kernel_size=3, stride=1, padding=1,
                                     groups=dim * 3 // 2, bias=bias)
        self.project_out2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        # self.qkv2 = nn.Conv2d(dim//2, dim*3//2, kernel_size=1, bias=bias)
        # self.qkv_dwconv2 = nn.Conv2d(dim*3//2, dim*3//2, kernel_size=3, stride=1, padding=1, groups=dim*3//2, bias=bias)
        # self.project_out2 = nn.Conv2d(dim//2, dim//2, kernel_size=1, bias=bias)
        # self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.fourier = FouierConv(dim // 2)

    def forward(self, x):
        # x1 for restormer attention
        # x2 for our attention
        x1, x2 = x.chunk(2, dim=1)  # x: [b, c, h, w]; x1, x2: [b, c, h/2, w/2]

        # ----------------------------------------------------------restormer attention---------------------------------------------------------------------
        # b,c,h,w = x1.shape

        # qkv = self.qkv_dwconv(self.qkv(x1))
        # q,k,v = qkv.chunk(3, dim=1)

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)

        # out = (attn @ v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = self.project_out(out)
        out = self.fourier(x1)
        # ----------------------------------------------------------restormer attention---------------------------------------------------------------------

        # -------------------------------------------------------------our attention------------------------------------------------------------------------
        b, c, h, w = x2.shape  # c, h, w均为x中的一半

        qkv = self.qkv_dwconv2(self.qkv2(x2))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.avgpool2(k)
        v = self.avgpool2(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h_down w_down -> b head c (h_down w_down)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # q: [b head c (h w)]
        k = torch.nn.functional.normalize(k, dim=-1)  # k: [b head c (h_down w_down)]

        attn = (k.transpose(-2,
                            -1) @ q)  # [b head c (h w)] @ [b head (h_down w_down) c] = [b head (h_down w_down) (h w)]
        attn = attn.softmax(dim=-2)

        out2 = (v @ attn)  # [b head c (h_down w_down)] @ [b head (h_down w_down) (h w)] = [b head (h w)]

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out2 = self.project_out2(out2)
        # -------------------------------------------------------------our attention------------------------------------------------------------------------

        return torch.cat([out, out2], dim=1)
        # return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.fourier = ResBlock_fft_bench(dim)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = x + self.fourier(self.norm2(x))
        x = x + self.ffn(self.norm3(x))

        return x

######################## fusion model ##################################
# 融合模块
class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # 假设x1和x2的通道数相同
        fused = torch.cat([x1, x2], dim=1)  # 通道数翻倍
        fused = self.fuse(fused)  # 通道数通过1x1卷积减半，以匹配后续层的期望输入
        return fused

class INNFusion(nn.Module):
    def __init__(self, channels):
        super(INNFusion, self).__init__()
        self.fuse = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
                      nn.Conv2d(2 * channels, channels, 1, 1, 0))
    def forward(self, x1,x2):
        x = self.fuse(torch.cat([x1,x2],dim=1))
        return x
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
# UNet风格的下采样块
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

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

        self.down_sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_feat*2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # return self.body(x)
        return self.down_sample(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(n_feat//2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # return self.body(x)
        return self.up_sample(x)
class FeatureConcatAndRestore2D(nn.Module):
    def __init__(self, in_channels, num_inputs=4, out_channels=256):
        """
        in_channels: channels of each input feature map (assumed same for all inputs)
        num_inputs: how many tensors will be concatenated (default 4)
        out_channels: desired output channel count (e.g. 256 to match existing convout)
        """
        super(FeatureConcatAndRestore2D, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        self.out_channels = out_channels

        concat_dim = in_channels * num_inputs
        # project across channel dimension using Conv1d (kernel_size=1) applied to the flattened spatial dimension
        self.conv1 = nn.Conv1d(in_channels=concat_dim, out_channels=out_channels, kernel_size=1)
        # small transpose conv to mimic V1.0's transposed conv step (keeps seq_len unchanged with padding)
        self.trans_conv = nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, f1, f2, f3, f4):
        # expect f1..f4 to be (B, C, H, W) and all have same H,W
        B, C, H, W = f1.shape
        seq_len = H * W

        # concatenate along channel dim: (B, C*num_inputs, H, W)
        x = torch.cat([f1, f2, f3, f4], dim=1)
        # reshape to (B, concat_dim, seq_len)
        x = x.view(B, -1, seq_len)
        # conv1d projection: (B, out_channels, seq_len)
        x = self.conv1(x)
        # bring to (B, seq_len, out_channels) for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.activation(x)
        # back to (B, out_channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.trans_conv(x)
        # reshape back to (B, out_channels, H, W)
        x = x.view(B, self.out_channels, H, W)
        return x

##########################################################################
##---------- Restormer -----------------------
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
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(HAformerSpatialFrequency, self).__init__()

        # 为每个下采样级别初始化一个融合模块, 正确设置融合模块的输入和输出通道
        self.fusion_module1 = INNFusion(dim)  # FusionModule(dim * 2, dim)  # 输入为两个stem层的输出通道数之和，输出为原始维度dim
        self.fusion_module2 = INNFusion(dim * 2)  # FusionModule(dim * 4, dim * 2)  # 输入为第一次融合和下采样后的通道数之和，输出为dim * 2
        self.fusion_module3 = INNFusion(dim * 4)  # FusionModule(dim * 8, dim * 4)  # 输入为第二次融合和下采样后的通道数之和，输出为dim * 4
        self.fusion_module4 = INNFusion(dim * 8)
        # self.fusion_module1 = FusionModule(dim * 2, dim)  # 输入为两个stem层的输出通道数之和，输出为原始维度dim
        # self.fusion_module2 = FusionModule(dim * 4, dim * 2)  # 输入为第一次融合和下采样后的通道数之和，输出为dim * 2
        # self.fusion_module3 = FusionModule(dim * 8, dim * 4)  # 输入为第二次融合和下采样后的通道数之和，输出为dim * 4
        # self.fusion_module4 = FusionModule(dim * 16, dim * 8)
        # UNet风格的卷积块，用于delay期图像，保持空间维度不变
        self.conv_block_delay1 = UNetConvBlock(dim, dim, stride=1)  # 保持空间维度
        self.conv_block_delay2 = UNetConvBlock(dim*2, dim*2, stride=1)  # 保持空间维度
        self.conv_block_delay3 = UNetConvBlock(dim*4, dim*4, stride=1) # 保持空间维度
        self.conv_block_delay4 = UNetConvBlock(dim * 8, dim * 8, stride=1)
        # UNet风格的下采样块，用于delay期图像
        self.downsample_delay1 = UNetConvBlock(dim, dim * 2)  # 第一次下采样，通道数翻倍，空间减半
        self.downsample_delay2 = UNetConvBlock(dim * 2, dim * 4)  # 第二次下采样
        self.downsample_delay3 = UNetConvBlock(dim * 4, dim * 8)  # 第三次下采样

        # Conv stem for art and pv
        self.stem_art_pv = OverlapPatchEmbed(2, dim)  # 输入通道为2，因为是art和pv concat在一起
        # Conv stem for delay
        self.stem_delay = OverlapPatchEmbed(1, dim)  # 输入通道为1，只有delay

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_ap_1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down_ap_2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down_ap_3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        #
        # self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        #
        # self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        #
        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #
        # self.refinement = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.up1 = Up(in_channels=128, out_channels=64, layers=1)
        self.up2 = Up(in_channels=256, out_channels=64, layers=2)
        self.up3 = Up(in_channels=512, out_channels=64, layers=3)

        self.convout = nn.Conv2d(256,3,kernel_size=3, stride=1, padding=1, bias=bias)
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_artpv = nn.Conv2d(2, out_channels, kernel_size=1, bias=bias)

        # ===== Minimal integration point: bring the author's "FeatureConcatAndRestore" idea in a
        # small, self-contained 2D-friendly form. This class is intentionally lightweight and
        # only used to replace the final concatenation -> convout pipeline with a projected
        # concatenation + 1D conv projection (very similar in spirit to V1.0's approach).
        self.feature_concat2d = FeatureConcatAndRestore2D(in_channels=dim, num_inputs=4, out_channels=256)

    # @torch.cuda.amp.autocast()
    def forward(self, art, pv, delay):
        # Concatenate art and pv phases and pass them through the stem convolution
        art_pv = torch.cat([art, pv], dim=1)  # shape: (b, 2, 256, 256)
        feat_art_pv = self.stem_art_pv(art_pv)  # shape: (b, c, 256, 256)

        # Pass the delay phase through its own stem convolution
        feat_delay = self.stem_delay(delay)  # shape: (b, c, 256, 256)
   ########## 1 stage #############################################################
        # encoder and fusion  1 stage
        out_enc_level1_delay = self.conv_block_delay1(feat_delay)  # shape: (b, dim, 256, 256)
        out_enc_level1_art_pv = self.encoder_level1(feat_art_pv)  # shape: (b, dim, 256, 256)
        fused_level1 = self.fusion_module1(out_enc_level1_art_pv, out_enc_level1_delay)

        # Downsample  1 stage
        art_pv_downsampled1 = self.down_ap_1_2(fused_level1 + out_enc_level1_art_pv)
        delay_downsampled1 = self.downsample_delay1(fused_level1 + out_enc_level1_delay)

    ########## 2 stage #############################################################
        # encoder and fusion  2 stage
        out_enc_level2_delay = self.conv_block_delay2(delay_downsampled1)
        out_enc_level2_art_pv = self.encoder_level2(art_pv_downsampled1)
        fused_level2 = self.fusion_module2(out_enc_level2_art_pv, out_enc_level2_delay)

        # Downsample 2 stage
        art_pv_downsampled2 = self.down_ap_2_3(fused_level2+out_enc_level2_art_pv)
        delay_downsampled2 = self.downsample_delay2(fused_level2 + out_enc_level2_delay)

    ########## 3 stage #############################################################
        # encoder and fusion  3 stage
        out_enc_level3_delay = self.conv_block_delay3(delay_downsampled2)
        out_enc_level3_art_pv = self.encoder_level3(art_pv_downsampled2)
        fused_level3 = self.fusion_module3(out_enc_level3_art_pv, out_enc_level3_delay)

        # Downsample 3 stage
        art_pv_downsampled3 = self.down_ap_3_4(fused_level3 + out_enc_level3_art_pv)
        delay_downsampled3 = self.downsample_delay3(fused_level3 + out_enc_level3_delay)

    ########## 4 stage #############################################################
        # encoder and fusion
        out_enc_level4_delay = self.conv_block_delay4(delay_downsampled3)
        out_enc_level4_art_pv = self.latent(art_pv_downsampled3)
        fusion_level4 = self.fusion_module4(out_enc_level4_art_pv, out_enc_level4_delay)

    ########## upsampling stage #############################################################

        up1 = self.up1(fused_level2)
        up2 = self.up2(fused_level3)
        up3 = self.up3(fusion_level4)
        # original: output = torch.cat([up1, up2, up3, fused_level1], 1)
        # original: output = self.convout(output)
        # Minimal change: use the user's feature-concat idea adapted to 2D
        # to combine the four tensors before the final conv.
        concat_feat = self.feature_concat2d(up1, up2, up3, fused_level1)
        output = self.convout(concat_feat)

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


# ----------------- Helper: adapted FeatureConcatAndRestore2D ---------------------
# This class is a small, 2D-friendly adaptation of the FeatureConcatAndRestore idea from V1.0.
# It concatenates N feature maps along channel dimension, projects with a 1x1 (as Conv1d) and
# a small transpose conv, then reshapes back to (B, C_out, H, W). The module is intentionally
# self-contained so the rest of the code remains unchanged.

