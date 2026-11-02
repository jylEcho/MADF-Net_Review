import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, num_classes=3):
                super(UNet, self).__init__()

                # 收缩路径
                self.down_conv1 = self.conv_stage(1, 64)
                self.down_conv2 = self.conv_stage(64, 128)
                self.down_conv3 = self.conv_stage(128, 256)
                self.down_conv4 = self.conv_stage(256, 512)

                # 扩展路径（注意通道数的改变）
                self.up_conv3 = self.conv_stage(512 + 256, 256)  # 由于拼接，输入通道数变为768
                self.up_conv2 = self.conv_stage(256 + 128, 128)
                self.up_conv1 = self.conv_stage(128 + 64, 64)

                self.final_conv = nn.Conv2d(64, num_classes, 1)  # 假设有两个类别

    def conv_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 收缩路径
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(F.max_pool2d(x1, 2))
        x3 = self.down_conv3(F.max_pool2d(x2, 2))
        x4 = self.down_conv4(F.max_pool2d(x3, 2))

        # 扩展路径
        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv1(x)

        x = self.final_conv(x)
        return x

if __name__ == '__main__':

    # 创建模型实例
    unet = UNet()

    # 假设输入图像的尺寸为 224x224
    input_tensor = torch.rand(1, 1, 224, 224)  # 1个通道的图像

    # 获取输出
    output = unet(input_tensor)
    print(output.shape)  # 输出尺寸
