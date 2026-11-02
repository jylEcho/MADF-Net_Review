import torch
import torch.nn as nn
from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
from swin_transformer import SwinTransformer   # 你需要提供实现

# ====================== Fusion Block (接口符合你要求) ======================
class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FusionBlock, self).__init__()
        self.proj = nn.Conv2d(in_dim * 4, out_dim, kernel_size=1)

    def forward(self, art_feat, pv_feat, dl_feat, fusion_feat, pos_enc=None):
        # 这里假设四路特征空间大小一致
        x = torch.cat([art_feat, pv_feat, dl_feat, fusion_feat], dim=1)
        out = self.proj(x)
        # 返回三个分支 (可按需调整)
        return out, out, out


# ====================== 主体网络 ======================
class MultiModalFusionNet(nn.Module):
    def __init__(self, img_size=224, embed_dim=96, out_channels=3):
        super(MultiModalFusionNet, self).__init__()

        # 共享 Swin-Transformer 编码器
        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,              # 我们会在 forward 中调整单通道输入
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

        # 融合模块
        self.fusion_block = FusionBlock(in_dim=embed_dim, out_dim=embed_dim)

        # 上采样部分
        self.up1 = Up(in_channels=embed_dim*2, out_channels=embed_dim, layers=1)
        self.up2 = Up(in_channels=embed_dim*4, out_channels=embed_dim, layers=2)
        self.up3 = Up(in_channels=embed_dim*8, out_channels=embed_dim, layers=3)

        # 最终输出卷积
        self.convout = nn.Conv2d(embed_dim*4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, art, pv, dl, fusion):
        B = art.shape[0]

        # =========== 共享编码器提取特征 ===========
        # 因为 art/pv/dl 是单通道，我们扩展到 3 通道输入
        art_feat = self.encoder(art.repeat(1,3,1,1))     # (B,C,H,W)
        pv_feat  = self.encoder(pv.repeat(1,3,1,1))
        dl_feat  = self.encoder(dl.repeat(1,3,1,1))
        fusion_feat = self.encoder(fusion)               # (B,C,H,W)

        # =========== 融合模块 ===========
        art_out, pv_out, dl_out = self.fusion_block(art_feat, pv_feat, dl_feat, fusion_feat)

        # =========== 上采样解码 ===========
        up1 = self.up1(art_out)
        up2 = self.up2(pv_out)
        up3 = self.up3(dl_out)

        out = torch.cat([up1, up2, up3, fusion_feat], dim=1)
        out = self.convout(out)

        return out
