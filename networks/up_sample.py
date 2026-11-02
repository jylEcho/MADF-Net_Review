import torch
import torch.nn as nn


class FlexibleDeconvUpsampleModuleWithBN(nn.Module):
    def __init__(self, in_channels, out_channels=64, layers=1):
        super(FlexibleDeconvUpsampleModuleWithBN, self).__init__()
        self.layers = layers
        modules = []

        # 对于每一层上采样，我们添加一个转置卷积层，后面跟着一个批量归一化层和ReLU激活层
        for i in range(layers):
            if i == layers - 1:
                # 最后一层调整为目标通道数
                current_out_channels = out_channels
            else:
                # 中间层保持通道数不变
                current_out_channels = in_channels

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=current_out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            modules.append(nn.BatchNorm2d(current_out_channels))
            modules.append(nn.ReLU(True))
            # 更新下一层的输入通道数
            in_channels = current_out_channels

        self.upsample_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsample_layers(x)

if __name__ == "__main__":
    # 示例使用
    upsample_once = FlexibleDeconvUpsampleModuleWithBN(in_channels=128, out_channels=64, layers=1)
    upsample_twice = FlexibleDeconvUpsampleModuleWithBN(in_channels=256, out_channels=64, layers=2)
    upsample_thrice = FlexibleDeconvUpsampleModuleWithBN(in_channels=512, out_channels=64, layers=3)

    # 假定输入张量
    input_tensor_1 = torch.randn(1, 128, 128, 128)
    input_tensor_2 = torch.randn(1, 256, 64, 64)
    input_tensor_3 = torch.randn(1, 512, 32, 32)

    # 上采样操作
    output_tensor_1 = upsample_once(input_tensor_1)
    output_tensor_2 = upsample_twice(input_tensor_2)
    output_tensor_3 = upsample_thrice(input_tensor_3)

    print("Output shape after 1 upsample layer with BN:", output_tensor_1.size())
    print("Output shape after 2 upsample layers with BN:", output_tensor_2.size())
    print("Output shape after 3 upsample layers with BN:", output_tensor_3.size())
