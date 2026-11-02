import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np
# from .modules import *
# from .modules_PA import *
# from networks.modules import CAM_Module, PA_Module
import torch
import math
from torch.nn import Module, Conv2d, Parameter,Sigmoid, Softmax
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'PA_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PA_Module(Module):
    """ Phase attention module"""
    def __init__(self, in_dim):
        super(PA_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()
        self.softmax11 = Softmax()
    def forward(self,x_fix,x_ref):
        """ calculate dependency among x_fix and x_ref, adaptively select x_ref according to the correlation of x_ref and x_fix
            inputs :
                x_fix : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x_ref.size()
        proj_ref = x_ref.view(m_batchsize, C, -1)
        proj_fix = x_fix.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_ref, proj_fix)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        channel_scale = torch.sum(attention, 2)
        channel_scale = channel_scale.unsqueeze(-1)
        channel_scale = self.sigmoid(channel_scale)

        proj_value = x_ref.view(m_batchsize, C, -1)  # reshape
        out = channel_scale * proj_value
        out = out.view(m_batchsize, C, height, width)  # reshape
        out = self.gamma * out + x_ref

        return out
class ResBlock(nn.Module):

    def __init__(self, nin, nout):
        super().__init__()

        self.bn =  nn.Sequential(
            #nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(nin, nout, 3, padding = 1), #conv1 in bottleneck
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nout, nout, 3, padding = 1),
            nn.BatchNorm2d(nout),
        )

        self.residual = nn.Sequential()

        if nin != nout:
            self.residual = nn.Sequential(
                nn.Conv2d(nin,nout,kernel_size=1,bias=False),
                nn.BatchNorm2d(nout),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bn(x)+self.residual(x))

class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv =  nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding = 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv(x)

class ResEncoder(nn.Module):
    def __init__(self, num_channels, nc):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )

        self.second_1 = ResBlock(nc, nc*2)
        self.second_2 = ResBlock(nc*2, nc*2)

        self.third_1 = ResBlock(nc*2, nc*4)
        self.third_2 = ResBlock(nc*4, nc*4)

        self.fourth_1 = ResBlock(nc*4, nc*8)
        self.fourth_2 = ResBlock(nc*8, nc*8)

        self.fifth_1 = ResBlock(nc*8,nc*8)
        self.fifth_2 = ResBlock(nc*8,nc*8)

        self.sixth_1 = ResBlock(nc*8,nc*8)
        self.sixth_2 = ResBlock(nc*8,nc*8)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)


    def forward(self, x):
        scale1 = self.first(x)
        scale2 = self.second_2(self.second_1(self.pool(scale1)))
        scale3 = self.third_2(self.third_1(self.pool(scale2)))
        scale4 = self.fourth_2(self.fourth_1(self.pool(scale3)))
        scale5 = self.fifth_2(self.fifth_1(self.pool(scale4)))
        scale6 = self.sixth_2(self.sixth_1(self.pool(scale5)))

        return scale1,scale2,scale3,scale4,scale5,scale6

class FANetHead(nn.Module):  # fusion with phase attention
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FANetHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv5p = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)  # intra-phase attention
        self.pa = PA_Module(inter_channels)  # inter-phase attention
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

    def forward(self, x_pv, x_art):
        feat1 = self.conv5c(x_pv)
        sc_feat = self.sc(feat1)
        sc_conv = self.conv51(sc_feat)

        feat2 = self.conv5p(x_art)
        pa_feat = self.pa(feat1,feat2)
        pa_conv = self.conv52(pa_feat)

        output = torch.cat([sc_conv, pa_conv], 1)

        return output

class RESNET_2D(nn.Module):  # 2D-Resnet

    def __init__(self, num_classes, num_channels, nc):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )

        self.second_1 = ResBlock(nc, nc*2)
        self.second_2 = ResBlock(nc*2, nc*2)

        self.third_1 = ResBlock(nc*2, nc*4)
        self.third_2 = ResBlock(nc*4, nc*4)

        self.fourth_1 = ResBlock(nc*4, nc*8)
        self.fourth_2 = ResBlock(nc*8, nc*8)

        self.fifth_1 = ResBlock(nc*8,nc*8)
        self.fifth_2 = ResBlock(nc*8,nc*8)

        self.sixth_1 = ResBlock(nc*8,nc*8)
        self.sixth_2 = ResBlock(nc*8,nc*8)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.up1 = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up2 = self.up_layer(nin=nc*2,nout=16,num_up=1)
        self.up3 = self.up_layer(nin=nc*4,nout=16,num_up=2)
        self.up4 = self.up_layer(nin=nc*8,nout=16,num_up=3)
        self.up5 = self.up_layer(nin=nc*8,nout=16,num_up=4)
        self.up6 = self.up_layer(nin=nc*8,nout=16,num_up=5)

        self.final = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,num_classes,kernel_size=1,padding=0),
        )

        #self._initialize_weights()

    def up_layer(self,nin, nout, num_up):
        cur_in = nin
        cur_out = ((2 ** (num_up - 1)) * nout)

        layers = []
        for i in range(num_up):
            layers.append(Deconv(cur_in, cur_out))
            cur_in = cur_out
            cur_out = math.floor(cur_out / 2)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # print(m.bias)
                # if m.bias:
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        scale1 = self.first(x)
        scale2 = self.second_2(self.second_1(self.pool(scale1)))
        scale3 = self.third_2(self.third_1(self.pool(scale2)))
        scale4 = self.fourth_2(self.fourth_1(self.pool(scale3)))
        scale5 = self.fifth_2(self.fifth_1(self.pool(scale4)))
        scale6 = self.sixth_2(self.sixth_1(self.pool(scale5)))

        up1 = self.up1(scale1)
        up2 = self.up2(scale2)
        up3 = self.up3(scale3)
        up4 = self.up4(scale4)
        up5 = self.up5(scale5)
        up6 = self.up6(scale6)

        concat_up = torch.cat([up1,up2,up3,up4,up5,up6],1)

        return self.final(concat_up)


class RESNET_phase_att(nn.Module):  # phase attention 20200414
    def __init__(self, num_classes, num_channels, nc):
        super().__init__()
        self.PV_encoder = ResEncoder(num_channels, nc)
        self.ART_encoder = ResEncoder(num_channels, nc)


        self.cf1 = FANetHead(nc)
        self.cf2 = FANetHead(nc*2)
        self.cf3 = FANetHead(nc*4)
        self.cf4 = FANetHead(nc*8)
        self.cf5 = FANetHead(nc*8)
        self.cf6 = FANetHead(nc*8)

        self.up1 = nn.Sequential(
            nn.Conv2d(int(nc * 2/4), 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up2 = self.up_layer(nin=int(nc * 2 * 2/4), nout=16, num_up=1)
        self.up3 = self.up_layer(nin=int(nc * 4 * 2/4), nout=16, num_up=2)
        self.up4 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=3)
        self.up5 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=4)
        self.up6 = self.up_layer(nin=int(nc * 8 * 2/4), nout=16, num_up=5)


        self.final = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, num_classes, kernel_size=1, padding=0),
        )
        # self._initialize_weights()

    def up_layer(self, nin, nout, num_up):
        cur_in = nin
        cur_out = ((2 ** (num_up - 1)) * nout)

        layers = []
        for i in range(num_up):
            layers.append(Deconv(cur_in, cur_out))
            cur_in = cur_out
            cur_out = math.floor(cur_out / 2)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # print(m.bias)
                # if m.bias:
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def forward(self, x_pv, x_art):
    def forward(self, x_art, x_pv, x_dl):
        scale1, scale2, scale3, scale4, scale5, scale6 = self.PV_encoder(x_pv)
        c1, c2, c3, c4, c5, c6 = self.ART_encoder(x_art)

        f1 = self.cf1(scale1, c1)
        f2 = self.cf2(scale2, c2)
        f3 = self.cf3(scale3, c3)
        f4 = self.cf4(scale4, c4)
        f5 = self.cf5(scale5, c5)
        f6 = self.cf6(scale6, c6)

        up1 = self.up1(f1)
        up2 = self.up2(f2)
        up3 = self.up3(f3)
        up4 = self.up4(f4)
        up5 = self.up5(f5)
        up6 = self.up6(f6)

        concat_up = torch.cat([up1, up2, up3, up4, up5, up6], 1)

        return self.final(concat_up)

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
###############################################################################
# if __name__ == '__main__':
#     import os
#     from thop import profile
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     model = RESNET_phase_att(num_classes=3, num_channels=1, nc=64).cuda()
#     x = torch.randn(1, 1, 224, 224).cuda()

#     # 计算 FLOPs 和参数量
#     flops, params = profile(model, inputs=(x, x))
#     print(f"FLOPs: {flops / 1e9} G FLOPs")
#     print(f"模型参数数量: {params / 1e6} M")

#     # 实际前向传播
#     output = model(x, x)
#     print("\n最终输出形状:", output.shape)
#     print("最终输出值:\n", output)
###############################################################################
if __name__ == '__main__':
    from thop import profile

    # 根据实际配置修改 num_classes 和 nc
    net = RESNET_phase_att(
        num_classes=3,
        num_channels=1,
        nc=64
    )
    net.eval()

    x_art = torch.randn(1, 1, 224, 224)
    x_pv = torch.randn(1, 1, 224, 224)
    x_dl = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        output = net(x_art, x_pv, x_dl)

    flops, _ = profile(
        net,
        inputs=(x_art, x_pv, x_dl),
        verbose=False
    )

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(
        p.numel() for p in net.parameters()
        if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print('=' * 45)
    print(f'Output Shape      : {tuple(output.shape)}')
    print(f'FLOPs             : {flops / 1e9:.4f} G')
    print(f'Total Params      : {total_params / 1e6:.4f} M')
    print(f'Trainable Params  : {trainable_params / 1e6:.4f} M')
    print(f'Frozen Params     : {frozen_params / 1e6:.4f} M')
    print(f'Trainable Ratio   : {trainable_params / total_params * 100:.2f}%')
    print(f'Model Size (FP32) : {total_params * 4 / 1024**2:.2f} MB')
    print('=' * 45)
###############################################################################