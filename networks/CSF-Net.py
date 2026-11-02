# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'networks')))
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .fusion_block import MultiEpisodeFusion
from .fusion_block import MultiEpisodeFusionBlock
from .fusion_block import RoPE


# from vmamba import VSSBlock
# from functools import partial

logger = logging.getLogger(__name__)

class FeatureConcatAndRestore(nn.Module):
    def __init__(self, hidden_dim=768, num_patches=49, output_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # 计算拼接后的维度 (4个特征图拼接)
        concat_dim = hidden_dim * 4

        # 1x1卷积用于降维，将拼接后的维度恢复为原始维度
        self.conv1 = nn.Conv1d(
            in_channels=concat_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 卷积层用于调整特征图大小
        self.trans_conv = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 归一化和激活函数
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, f1, f2, f3, f4):
        """
        输入: 4个单独的特征图，每个形状为 (16, 49, 768)
        输出: 恢复后的特征图，形状为 (16, 49, 768)
        """
        # 确保所有特征图尺寸一致
        for i, f in enumerate([f1, f2, f3, f4]):
            assert f.shape == (16, 49, 768), \
                f"第{i + 1}个特征图尺寸不正确，应为(16, 49, 768)，实际为{f.shape}"

        # 将4个特征图在通道维度拼接 (16, 49, 768*4) = (16, 49, 3072)
        concatenated = torch.cat([f1, f2, f3, f4], dim=2)

        # 调整维度以适应卷积操作: (16, 3072, 49)
        x = concatenated.permute(0, 2, 1)

        # 1x1卷积降维到原始维度: (16, 768, 49)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # 恢复维度顺序: (16, 49, 768)
        x = self.norm(x)
        x = self.activation(x)

        # 转置卷积调整大小
        x = x.permute(0, 2, 1)  # (16, 768, 49)
        x = self.trans_conv(x)
        x = x.permute(0, 2, 1)  # 恢复维度顺序: (16, 49, 768)

        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet1 = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


        self.swin_unet2 = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


        self.swin_unet3 = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


        self.swin_unet4 = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

        self.fusion_block = MultiEpisodeFusionBlock(
            dim=768,
            ssm_ratio=1.0,
            exp_ratio=4.0,
            inner_kernel_size=3,
            num_heads=24,  
            use_rpb=True,
            drop_path=0.1
        )

        self.rope = RoPE(embed_dim=768, num_heads=24)
        self.pos_enc = self.rope(slen=(7, 7))

        # self.pos_enc = (self.pos_enc[0].to(device), self.pos_enc[1].to(device))

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 或者如果 args 中有设备参数，可以使用
        # self.device = args.device
        
        # 然后修改原来的代码，使用 self.device
        self.pos_enc = (self.pos_enc[0].to(self.device), self.pos_enc[1].to(self.device))


        self.feature_concat = FeatureConcatAndRestore(
            hidden_dim=96,  
            output_dim=768   
        )
 
        hidden_dim = 768  # 隐藏层维度
        drop_path = 0.  # 随机深度概率
        attn_drop_rate = 0.  # 注意力 dropout 概率
        d_state = 16  # 状态维度

        # 定义归一化层（VSSBlock需要）
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 实例化VSSBlock
        # self.vss_block = VSSBlock(
        #     hidden_dim=hidden_dim,
        #     drop_path=drop_path,
        #     norm_layer=norm_layer,
        #     attn_drop_rate=attn_drop_rate,
        #     d_state=d_state
        # )


        self.Conv21 = conv_block(1536, 768)
        self.Conv31 = conv_block(3072, 1536)
        self.Conv32 = conv_block(1536, 768)
        self.Conv41 = conv_block(6144, 3072)
        self.Conv42 = conv_block(3072, 1536)
        self.Conv43 = conv_block(1536, 768)

    def reshape(self, encoder_output):
        spatial_size = int(encoder_output.shape[1] ** 0.5)
        batch_size, seq_len, hidden_dim = encoder_output.shape
        restored_tensor = encoder_output.view(batch_size, hidden_dim, spatial_size, spatial_size)
        return restored_tensor

    def shape(self, tensor):
        batch_size, hidden_dim, spatial_size, _ = tensor.shape
        seq_len = spatial_size * spatial_size
        flattened = tensor.view(batch_size, hidden_dim, seq_len)
        restored_tensor = flattened.permute(0, 2, 1)
        return restored_tensor



    def forward(self, image_batch_a, image_batch_v, image_batch_d):
        if image_batch_a.size()[1] == 1:
            image_batch_a_3 = image_batch_a.repeat(1, 3, 1, 1)
        if image_batch_v.size()[1] == 1:
            image_batch_v_3 = image_batch_v.repeat(1, 3, 1, 1)
        if image_batch_d.size()[1] == 1:
            image_batch_d_3 = image_batch_d.repeat(1, 3, 1, 1)

        image_batch_adv=torch.cat((image_batch_a,image_batch_v,image_batch_d),dim=1)

        # encoder_output_a, x_downsample_a = self.swin_unet1.forward_features(image_batch_a)
        # encoder_output_v, x_downsample_v = self.swin_unet2.forward_features(image_batch_v)
        # encoder_output_d, x_downsample_d = self.swin_unet3.forward_features(image_batch_d)
        # encoder_output_adv, x_downsample_adv = self.swin_unet4.forward_features(image_batch_d)
        encoder_output_a, x_downsample_a = self.swin_unet1.forward_features(image_batch_a_3)
        # print(f"encoder_output_a shape: {encoder_output_a.shape}")
        encoder_output_v, x_downsample_v = self.swin_unet2.forward_features(image_batch_v_3)
        # print(f"encoder_output_v shape: {encoder_output_v.shape}")
        encoder_output_d, x_downsample_d = self.swin_unet3.forward_features(image_batch_d_3)
        # print(f"encoder_output_d shape: {encoder_output_d.shape}")
        encoder_output_adv, x_downsample_adv = self.swin_unet4.forward_features(image_batch_adv)
        # print(f"encoder_output_adv shape: {encoder_output_adv.shape}")

        encoder_output_a_fusion = torch.cat((encoder_output_a, encoder_output_v), dim=2)
        # print(f"encoder_output_a_fusion shape: {encoder_output_a_fusion.shape}")
        encoder_output_d_fusion = torch.cat((encoder_output_d,encoder_output_a_fusion, encoder_output_v), dim=2)
        # print(f"encoder_output_d_fusion shape: {encoder_output_d_fusion.shape}")
        encoder_output_v_fusion = encoder_output_v
        # print(f"encoder_output_v_fusion shape: {encoder_output_v_fusion.shape}")
        encoder_output_adv_fusion =torch.cat((encoder_output_adv,encoder_output_a_fusion, encoder_output_d_fusion, encoder_output_v_fusion), dim=2)
        # print(f"encoder_output_adv_fusion shape: {encoder_output_adv_fusion.shape}")

        encoder_output_a_reshapped = self.reshape(encoder_output_a_fusion)
        # print(f"encoder_output_a_reshapped shape: {encoder_output_a_reshapped.shape}")
        encoder_output_d_reshapped = self.reshape(encoder_output_d_fusion)
        # print(f"encoder_output_d_reshapped shape: {encoder_output_d_reshapped.shape}")
        encoder_output_v_reshapped = self.reshape(encoder_output_v_fusion)
        # print(f"encoder_output_v_reshapped shape: {encoder_output_v_reshapped.shape}")
        encoder_output_adv_reshapped = self.reshape(encoder_output_adv_fusion)
        # print(f"encoder_output_adv_reshapped shape: {encoder_output_adv_reshapped.shape}")

        conv_a = self.Conv21(encoder_output_a_reshapped)
        # print(f"conv_a shape: {conv_a.shape}")
        conv_d1 = self.Conv31(encoder_output_d_reshapped)
        conv_d = self.Conv32(conv_d1)
        # print(f"conv_d shape: {conv_d.shape}")
        conv_adv1 = self.Conv41(encoder_output_adv_reshapped)
        conv_adv2 = self.Conv42(conv_adv1)
        conv_adv = self.Conv43(conv_adv2)
        # print(f"conv_adv shape: {conv_adv.shape}")

        # conv_a_shape = self.shape(conv_a)
        # print(f"conv_a_shape shape: {conv_a_shape.shape}")
        # conv_d_shape = self.shape(conv_d)
        # print(f"conv_d_shape shape: {conv_d_shape.shape}")
        # conv_adv_shape = self.shape(conv_adv)
        # print(f"conv_adv_shape shape: {conv_adv_shape.shape}")
        # print(f"conv_a shape: {conv_a.shape}")
        # print(f"encoder_output_v_reshapped shape: {encoder_output_v_reshapped.shape}")
        # print(f"conv_d shape: {conv_d.shape}")
        # print(f"conv_adv shape: {conv_adv.shape}")
        # print(f"self.pos_enc shape: {self.pos_enc.shape}")

        art_out, pv_out, dl_out = self.fusion_block(conv_a, encoder_output_v_reshapped, conv_d, conv_adv, self.pos_enc)
        # art_out, pv_out, dl_out = MultiEpisodeFusion(conv_a, encoder_output_v_reshapped, conv_d, conv_adv, pos_enc)
        # print(f"art_out shape: {art_out.shape}")
        # print(f"pv_out shape: {pv_out.shape}")
        # print(f"dl_out shape: {dl_out.shape}")

        pv_out_shape = self.shape(pv_out)
        # print(f"pv_out_shape shape: {pv_out_shape.shape}")
        x = self.swin_unet4.forward_up_features(pv_out_shape, x_downsample_v)

        x = self.swin_unet4.up_x4(x)
        # print(f"x shape: {x.shape}")

        return x


    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


if __name__ == '__main__':
    import time
    from torch.profiler import profile, record_function, ProfilerActivity


    class DummyConfig:
        class MODEL:
            PRETRAIN_CKPT = None  

        class TRAIN:
            BATCH_SIZE = 16
            IMG_SIZE = 224


    config = DummyConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    model = SwinUnet(config=config).to(device)
    model.feature_concat = FeatureConcatAndRestore(hidden_dim=768, num_patches=49).to(device)

    print("\n===== 模型结构概览 =====")
    print(model)

    print("\n===== 输入验证 =====")
    batch_size = config.TRAIN.BATCH_SIZE
    img_size = config.TRAIN.IMG_SIZE

    image_batch_a = torch.rand(batch_size, 1, img_size, img_size).to(device)
    image_batch_v = torch.rand(batch_size, 1, img_size, img_size).to(device)
    image_batch_d = torch.rand(batch_size, 1, img_size, img_size).to(device)

    print(f"输入A尺寸: {image_batch_a.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")
    print(f"输入V尺寸: {image_batch_v.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")
    print(f"输入D尺寸: {image_batch_d.shape} (预期: [{batch_size}, 1, {img_size}, {img_size}])")

    print("\n===== 前向传播测试（评估模式） =====")
    model.eval()
    with torch.no_grad():  
        start_time = time.time()
        output = model(image_batch_a, image_batch_v, image_batch_d)
        forward_time = time.time() - start_time

    print(f"输出尺寸: {output.shape} (预期: [{batch_size}, 3, {img_size}, {img_size}])")  
