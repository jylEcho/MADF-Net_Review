import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

# ----------------------------
# Fusion Block 示例
# ----------------------------
class FusionBlock(nn.Module):
    def __init__(self, embed_dim):
        super(FusionBlock, self).__init__()
        self.proj = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)

    def forward(self, art_out, pv_out, dl_out, fusion_out, pos_enc=None):
        # 假设四个分支的输出尺寸一致
        fused = torch.cat([art_out, pv_out, dl_out, fusion_out], dim=1)
        fused = self.proj(fused)
        return fused


# ----------------------------
# 总体模型
# ----------------------------
class MultiInputFusionModel(nn.Module):
    def __init__(self, img_size=224, embed_dim=96):
        super(MultiInputFusionModel, self).__init__()

        # stem，把输入投影到 embed_dim 通道
        self.stem_1ch = nn.Conv2d(1, embed_dim, kernel_size=3, stride=1, padding=1)
        self.stem_3ch = nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1)

        # 共享 Swin Transformer backbone
        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
        )

        # Fusion block
        self.fusion_block = FusionBlock(embed_dim)

        # 上采样部分（用你代码里的 Up 模块）
        from up_sample import FlexibleDeconvUpsampleModuleWithBN as Up
        self.up1 = Up(in_channels=embed_dim, out_channels=embed_dim // 2, layers=1)
        self.up2 = Up(in_channels=embed_dim // 2, out_channels=embed_dim // 4, layers=1)
        self.final_conv = nn.Conv2d(embed_dim // 4, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, art, pv, dl, fusion):
        # stem
        art_feat = self.stem_1ch(art)
        pv_feat = self.stem_1ch(pv)
        dl_feat = self.stem_1ch(dl)
        fusion_feat = self.stem_3ch(fusion)

        # 共享编码器
        art_out = self.encoder(art_feat)
        pv_out = self.encoder(pv_feat)
        dl_out = self.encoder(dl_feat)
        fusion_out = self.encoder(fusion_feat)

        # 融合
        fused = self.fusion_block(art_out, pv_out, dl_out, fusion_out)

        # 上采样恢复空间分辨率
        up1 = self.up1(fused)
        up2 = self.up2(up1)
        out = self.final_conv(up2)
        return out


if __name__ == "__main__":
    model = MultiInputFusionModel()
    art = torch.randn(1, 1, 224, 224)
    pv = torch.randn(1, 1, 224, 224)
    dl = torch.randn(1, 1, 224, 224)
    fusion = torch.randn(1, 3, 224, 224)

    out = model(art, pv, dl, fusion)
    print("输出 shape:", out.shape)  # (B,3,224,224)
