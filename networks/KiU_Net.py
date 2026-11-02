import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

class kiunet_org(nn.Module):

    def __init__(self,args):
        super(kiunet_org, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 2
        self.start = nn.Conv2d(in_ch, 3, 3, stride=2, padding=1)

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2= nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3= nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 = nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2= nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        # self.start = nn.Conv3d(1, 1, 3, stride=1, padding=1)
        self.final = nn.Conv2d(8,out_ch,1,stride=1,padding=0)
        self.fin = nn.Conv2d(out_ch,out_ch,1,stride=1,padding=0)

        self.map4 = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(8, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        # print(x.shape)
        outx = self.start(x)
        # print(outx.shape)
        out = F.relu(self.en1_bn(F.max_pool3d(self.encoder1(outx),2,2)))  #U-Net branch
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(outx),scale_factor=(0.5,1,1),mode ='trilinear'))) #Ki-Net branch
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(1,0.5,0.5),mode ='trilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(1,2,2),mode ='trilinear')) #CRFB

        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(self.en2_bn(F.max_pool3d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(1,1,1),mode ='trilinear')))
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.5,0.25,0.25),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(2,4,4),mode ='trilinear'))

        u2 = out
        o2 = out1
        out = F.pad(out,[0,0,0,0,0,1])
        # print(out.shape)
        out = F.relu(self.en3_bn(F.max_pool3d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(1,2,2),mode ='trilinear')))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.5,0.0625,0.0625),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(2,16,16),mode ='trilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear')))  #U-NET
        out1 = F.relu(self.def1_bn(F.max_pool3d(self.decoderf1(out1),2,2))) #Ki-NET
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(2,0.25,0.25),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(0.5,4,4),mode ='trilinear'))
        # print(out.shape)
        output1 = self.map4(out)
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(1,2,2),mode ='trilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool3d(self.decoderf2(out1),1,1)))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(1,0.5,0.5),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(1,2,2),mode ='trilinear'))
        output2 = self.map3(out)
        # print(out1.shape,o1.shape)
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(1,2,2),mode ='trilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool3d(self.decoderf3(out1),1,1)))
        # print(out.shape,out1.shape)
        output3 = self.map2(out)


        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.final(out))  #1*1 conv

        output4 = F.interpolate(self.fin(out),scale_factor=(4,4,4),mode ='trilinear')
        # print(out.shape)
        # out = self.soft(out)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)


class RecurrentBlock(nn.Module):
    def __init__(self, ch):
        super(RecurrentBlock, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return x * y + x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class KiU_Net(nn.Module):
    def __init__(self, args=None):
        super(KiU_Net, self).__init__()
        self.args = args

        in_channels = 1
        out_channels = 3

        self.conv1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = ConvBlock(256, 512)

        self.recursive_block1 = RecurrentBlock(32)
        self.recursive_block2 = RecurrentBlock(64)
        self.recursive_block3 = RecurrentBlock(128)
        self.recursive_block4 = RecurrentBlock(256)
        self.recursive_block5 = RecurrentBlock(512)

        self.up5 = UpConv(512, 256)
        self.att5 = AttentionBlock(256, 256, 128)
        self.dec5 = ConvBlock(512, 256)

        self.up4 = UpConv(256, 128)
        self.att4 = AttentionBlock(128, 128, 64)
        self.dec4 = ConvBlock(256, 128)

        self.up3 = UpConv(128, 64)
        self.att3 = AttentionBlock(64, 64, 32)
        self.dec3 = ConvBlock(128, 64)

        self.up2 = UpConv(64, 32)
        self.att2 = AttentionBlock(32, 32, 16)
        self.dec2 = ConvBlock(64, 32)

        self.outconv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c1 = self.recursive_block1(self.conv1(x))
        p1 = self.pool1(c1)

        c2 = self.recursive_block2(self.conv2(p1))
        p2 = self.pool2(c2)

        c3 = self.recursive_block3(self.conv3(p2))
        p3 = self.pool3(c3)

        c4 = self.recursive_block4(self.conv4(p3))
        p4 = self.pool4(c4)

        c5 = self.recursive_block5(self.conv5(p4))

        u5 = self.up5(c5, c4.shape[-2:])
        s4 = self.att5(u5, c4)
        d5 = self.dec5(torch.cat([s4, u5], dim=1))

        u4 = self.up4(d5, c3.shape[-2:])
        s3 = self.att4(u4, c3)
        d4 = self.dec4(torch.cat([s3, u4], dim=1))

        u3 = self.up3(d4, c2.shape[-2:])
        s2 = self.att3(u3, c2)
        d3 = self.dec3(torch.cat([s2, u3], dim=1))

        u2 = self.up2(d3, c1.shape[-2:])
        s1 = self.att2(u2, c1)
        d2 = self.dec2(torch.cat([s1, u2], dim=1))

        out = self.outconv(d2)
        return out

################################################################################
# if __name__ == "__main__":
#     from thop import profile
#     import torch

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = KiU_Net().to(device)
#     model.eval()
#     x = torch.rand((1, 1, 224, 224), device=device)

#     flops, params = profile(model, inputs=(x,), verbose=False)

#     print(f"FLOPs: {flops / 1e9:.6f} G FLOPs")
#     print(f"Params: {params / 1e6:.6f} M")
################################################################################
if __name__ == "__main__":
    from thop import profile
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = KiU_Net().to(device)
    model.eval()

    x = torch.rand(1, 1, 224, 224, device=device)

    # 前向测试
    with torch.no_grad():
        output = model(x)

    # FLOPs 统计
    flops, _ = profile(
        model,
        inputs=(x,),
        verbose=False
    )

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print("=" * 45)
    print(f"Device            : {device}")

    if isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
            print(f"Output[{i}] Shape   : {tuple(out.shape)}")
    else:
        print(f"Output Shape      : {tuple(output.shape)}")

    print(f"FLOPs             : {flops / 1e9:.6f} G")
    print(f"Total Params      : {total_params / 1e6:.6f} M")
    print(f"Trainable Params  : {trainable_params / 1e6:.6f} M")
    print(f"Frozen Params     : {frozen_params / 1e6:.6f} M")
    print(f"Trainable Ratio   : {trainable_params / total_params * 100:.2f}%")
    print(f"Model Size (FP32) : {total_params * 4 / 1024**2:.2f} MB")
    print("=" * 45)
################################################################################
