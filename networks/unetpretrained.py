import torch
import torch.nn as nn
import torchvision.models as models
import  torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 加载预训练的 ResNet 并使用其作为编码器
        pretrained_resnet = models.resnet18(pretrained=True)
        self.encoder_layers = list(pretrained_resnet.children())[:-2]  # 移除全连接层和最后的池化层
        self.encoder = nn.Sequential(*self.encoder_layers)

        # 解码器层
        self.up_conv3 = self.up_conv(512, 256)
        self.up_conv2 = self.up_conv(256 + 256, 128)  # 注意通道数的变化
        self.up_conv1 = self.up_conv(128 + 128, 64)

        self.final_conv = nn.Conv2d(64 + 64, 2, 1)  # 输出层

    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器路径
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1][1:]  # 逆序并去除最后一层

        # 解码器路径
        for i, up_conv in enumerate([self.up_conv3, self.up_conv2, self.up_conv1]):
            x = up_conv(x)
            skip_connection = skip_connections[i]
            if x.size() != skip_connection.size():
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip_connection], dim=1)

        x = self.final_conv(torch.cat([x, skip_connections[-1]], dim=1))  # 最终层
        return x


# 创建模型实例
unet = UNet()

# 假设输入尺寸
input_tensor = torch.rand(1, 3, 224, 224)  # 3个通道的图像

# 获取输出
output = unet(input_tensor)
print(output.shape)  # 输出尺寸
