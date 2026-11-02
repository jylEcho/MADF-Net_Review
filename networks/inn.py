import torch
import numpy as np
import scipy
import torch.nn as nn
import networks.thops as thops
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from networks.MSDNN import HinResBlock
# 假设你已经有了 InvertibleConv1x1 的定义
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False,device =torch.device('cpu')):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)).to(device))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)).to(device))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)).to(device))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)).to(device))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)).to(device))
            self.register_buffer('l_mask', torch.Tensor(l_mask).to(device))
            self.register_buffer('eye', torch.Tensor(eye).to(device))
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            device = input.device
            self.p = self.p.to(device)
            self.sign_s = self.sign_s.to(device)
            self.l_mask = self.l_mask.to(device)
            self.eye = self.eye.to(device)
            # self.l = self.l.cuda()
            # self.u = self.u.cuda()
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            # l = self.l * self.l_mask.to(input.device) + self.eye.to(input.device)
            # u = self.u * self.l_mask.to(input.device).transpose(0, 1).contiguous() + torch.diag(
            #     self.sign_s.to(input.device) * torch.exp(self.log_s))

            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def flow_permutation(self, z, logdet=0, rev=False):
        return self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out



class SimpleModel(nn.Module):
    def __init__(self, channels):
        super(SimpleModel, self).__init__()
        self.inv_conv = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
                                              nn.Conv2d(2*channels,channels,1,1,0))

    def forward(self, x):
        return self.inv_conv(x)

def main():
    # 设定基本参数
    num_channels = 6
    num_samples = 64
    image_size = 64

    # 检查是否有多个GPU可用
    if torch.cuda.device_count() < 1:
        print("This script requires at least 1 GPU.")
        return

    device = torch.device("cuda:0")

    # 创建一个简单的数据集
    inputs = torch.randn(num_samples, num_channels, image_size, image_size)
    targets = torch.randn(num_samples, num_channels, image_size, image_size)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=16)

    # 创建模型并移动到GPU
    model = SimpleModel(channels=num_channels).to(device)

    # 使用DataParallel包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 简单的训练循环
    model.train()
    for epoch in range(10):

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(torch.cat([inputs,inputs[:]],dim=1))
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    main()
