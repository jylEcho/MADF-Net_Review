import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv2(x)
        x = self.conv1(x)
        x += residual
        return x


class FusionBlock(nn.Module):
    def __init__(self, num_features, num_inputs):
        super(FusionBlock, self).__init__()
        self.num_features = num_features
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv1x1_list = nn.ModuleList(
            [nn.Conv2d(num_features, num_features, kernel_size=1) for _ in range(num_inputs)])
        self.softmax = nn.Softmax(dim=0)
        self.fusion_conv = nn.Conv2d(num_features, num_features, kernel_size=1)

    def forward(self, *inputs):
        # Compute weighted average
        weighted_features = [w * conv1x1(f) for w, conv1x1, f in
                             zip(self.softmax(self.weights), self.conv1x1_list, inputs)]
        weighted_sum = sum(weighted_features)

        # Compute the high-level feature map
        F_high = weighted_features[0]  # Assuming the first feature map is the high-level one
        F_other = weighted_sum - F_high

        # Attention mechanism
        attention = F_other * torch.sigmoid(F_high)

        # Fusion of the feature maps
        F_fusion = self.fusion_conv(attention)

        return F_fusion

class ModifiedUNet(nn.Module):
    def __init__(self):
        super(ModifiedUNet, self).__init__()
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            # ConvBlock(3, 32),
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        ])
        # Fusion blocks for each level of the encoder
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(256, 4),
            FusionBlock(256, 4),
            FusionBlock(192, 4),
            FusionBlock(96, 4)
        ])
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(96, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32)
        ])
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2)
        ])
        # Final output convolution
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        # Pooling layer for downsampling in the encoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # def forward(self, *inputs):    
    def forward(self, art, pv, dl):
        # Check the initial inputs
              
        inputs = [art, pv, dl]
        # print(f"Initial inputs shapes: {[x.shape for x in inputs]}")
        
        # Encoder path
        enc_features = []
        for idx, block in enumerate(self.encoder_blocks):
            inputs = [block(x) for x in inputs]
            # print(f"After Encoder Block {idx}, shapes: {[x.shape for x in inputs]}")
            if idx < len(self.encoder_blocks) - 1:
                # Save the current features for the skip connections
                enc_features.append(inputs)
                # Downsample for the next encoder block
                inputs = [self.pool(x) for x in inputs]

        # At this point, 'inputs' should be a list of feature maps from the last encoder block

        # Now we should apply each fusion block to its corresponding set of feature maps
        for idx, fusion_block in enumerate(self.fusion_blocks):
            # Fuse the feature maps; the result is a single feature map
            f_combined = fusion_block(*inputs)
            # 'f_combined' is now a single tensor

            # If not the last block, prepare inputs for the next fusion block
            if idx < len(self.fusion_blocks) - 1:
                # Upsample
                upsampled_input = self.upsample_layers[idx](f_combined)
                # Prepare for concatenation with skip connections from the encoder
                # We take the corresponding feature maps from 'enc_features' which is a list of lists of tensors
                inputs = [torch.cat((upsampled_input, enc_feature), dim=1)
                          for enc_feature in enc_features[-(idx + 1)]]                # After concatenation, 'inputs' should be back to being a list of tensors
            else:
                # If it's the last fusion block, we only have one tensor to work with
                inputs = [f_combined]

        # 'inputs' is now a list with a single tensor that went through all fusion blocks
        # We pass it through each decoder block
        for decoder_block in self.decoder_blocks:
            inputs = [decoder_block(inputs[0])]

        # Final convolution to get to the output
        output = self.final_conv(inputs[0])
        return output

if __name__ == '__main__':
    from thop import profile

    net = ModifiedUNet()
    net.eval()

    art = torch.randn(1, 1, 224, 224)
    pv = torch.randn(1, 1, 224, 224)
    dl = torch.randn(1, 1, 224, 224)

    # 前向测试
    with torch.no_grad():
        output = net(art, pv, dl)

    # FLOPs 和参数量
    flops, _ = profile(
        net,
        inputs=(art, pv, dl),
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