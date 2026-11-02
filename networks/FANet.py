import torch
import torch.nn as nn
if __package__:
    from .blocks import ResidualBlock, MixPool as _MixPool
else:
    from blocks import ResidualBlock, MixPool as _MixPool


class MixPool(_MixPool):
    """Device-safe implementation; parameter names match the original block."""
    def forward(self, x, masks):
        feature_mask = self.fmask(x) > 0.5
        masks = masks.to(device=x.device, dtype=x.dtype)
        masks = torch.nn.functional.adaptive_max_pool2d(masks, x.shape[-2:])
        gate = torch.logical_or(feature_mask, masks > 0.5).to(dtype=x.dtype)
        x1 = self.conv1(x * gate)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(EncoderBlock, self).__init__()

        self.name = name
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.r1(inputs)
        x = self.r2(x)
        p = self.p1(x, masks)
        o = self.pool(p)
        return o, x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        p = self.p1(x, masks)
        return p

class FANet(nn.Module):
    def __init__(self, args=None, *, in_channels=1, num_classes=None):
        super(FANet, self).__init__()
        # Supports FANet(), FANet(args), and explicit keyword construction.
        self.in_channels = int(getattr(args, "in_channels", in_channels))
        if num_classes is None:
            num_classes = getattr(args, "num_classes", 3)
        self.num_classes = int(num_classes)
        if self.in_channels < 1 or self.num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive")

        self.e1 = EncoderBlock(self.in_channels, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
        self.d4 = DecoderBlock(32, 16)

        self.output = nn.Conv2d(16+1, self.num_classes, kernel_size=1, padding=0)

    def forward(self, inputs, masks=None):
        # Single-pass adaptation: no external mask means a zero initial guide
        # in both training and inference. This is not iterative feedback training.
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2 or masks is not None:
                raise ValueError("Use images or [images, masks], without ambiguous arguments")
            inputs, masks = inputs
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} image channels, got {inputs.shape[1]}")
        if any(size < 16 or size % 16 for size in inputs.shape[-2:]):
            raise ValueError("Image height and width must be positive multiples of 16")
        expected = (inputs.shape[0], 1, *inputs.shape[-2:])
        if masks is None:
            masks = inputs.new_zeros(expected)
        else:
            if not isinstance(masks, torch.Tensor) or tuple(masks.shape) != expected:
                raise ValueError(f"masks must have shape {expected}")
            masks = masks.to(device=inputs.device, dtype=inputs.dtype)

        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)

        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)

        d5 = torch.cat([d4, masks], axis=1)
        output = self.output(d5)

        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FANet(in_channels=1, num_classes=3).to(device)
    x = torch.randn(2, 1, 224, 224, device=device)
    print(model(x).shape)  # [2, 3, 224, 224], raw logits
