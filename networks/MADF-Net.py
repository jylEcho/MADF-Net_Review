import torch
import torch.nn as nn
import torch.nn.functional as F


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


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-Attention Module
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))  # Scale factor

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W)  # [B, C', H*W]
        key = self.key(x).view(B, -1, H * W)  # [B, C', H*W]
        value = self.value(x).view(B, -1, H * W)  # [B, C, H*W]

        query = query.permute(0, 2, 1)  # [B, H*W, C']
        attention = torch.bmm(query, key)  # [B, H*W, H*W]
        attention = self.softmax(attention)  # [B, H*W, H*W]

        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)
        return self.gamma * out + x  # Residual connection


class LiNet(nn.Module):
    """
    UNet - Basic Implementation
    """
    def __init__(self, args):
        super(LiNet, self).__init__()
        self.args = args
        in_ch = 1
        out_ch = 3

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Self-Attention modules
        self.self_attn1 = SelfAttention(filters[0])
        self.self_attn2 = SelfAttention(filters[1])
        self.self_attn3 = SelfAttention(filters[2])
        self.self_attn4 = SelfAttention(filters[3])
        self.self_attn5 = SelfAttention(filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv111 = conv_block(in_ch, filters[0])
        self.Conv211 = conv_block(filters[0], filters[1])
        self.Conv311 = conv_block(filters[1], filters[2])
        self.Conv411 = conv_block(filters[2], filters[3])
        self.Conv511 = conv_block(filters[3], filters[4])
        self.Conv112 = conv_block(in_ch, filters[0])
        self.Conv212 = conv_block(filters[0], filters[1])
        self.Conv312 = conv_block(filters[1], filters[2])
        self.Conv412 = conv_block(filters[2], filters[3])
        self.Conv512 = conv_block(filters[3], filters[4])
        self.Conv113 = conv_block(in_ch, filters[0])
        self.Conv213 = conv_block(filters[0], filters[1])
        self.Conv313 = conv_block(filters[1], filters[2])
        self.Conv413 = conv_block(filters[2], filters[3])
        self.Conv513 = conv_block(filters[3], filters[4])
        self.Conv114 = conv_block(3, 32)
        self.Conv214 = conv_block(128, 64)
        self.Conv314 = conv_block(256, 128)
        self.Conv414 = conv_block(512, 256)
        self.Conv514 = conv_block(1024, 512)
        self.Conv115 = conv_block(128, 128)
        self.Conv215 = conv_block(256, 256)
        self.Conv315 = conv_block(512, 512)
        self.Conv415 = conv_block(1024, 1024)
        self.Conv515 = conv_block(2048, 2048)

        self.Up511 = up_conv(filters[4], filters[3])
        self.Up_conv511 = conv_block(filters[4], filters[3])
        self.Up411 = up_conv(filters[3], filters[2])
        self.Up_conv411 = conv_block(filters[3], filters[2])
        self.Up311 = up_conv(filters[2], filters[1])
        self.Up_conv311 = conv_block(filters[2], filters[1])
        self.Up211 = up_conv(filters[1], filters[0])
        self.Up_conv211 = conv_block(filters[1], filters[0])
        self.Up512 = up_conv(filters[4], filters[3])
        self.Up_conv512 = conv_block(filters[4], filters[3])
        self.Up412 = up_conv(filters[3], filters[2])
        self.Up_conv412 = conv_block(filters[3], filters[2])
        self.Up312 = up_conv(filters[2], filters[1])
        self.Up_conv312 = conv_block(filters[2], filters[1])
        self.Up212 = up_conv(filters[1], filters[0])
        self.Up_conv212 = conv_block(filters[1], filters[0])
        self.Up513 = up_conv(filters[4], filters[3])
        self.Up_conv513 = conv_block(filters[4], filters[3])
        self.Up413 = up_conv(filters[3], filters[2])
        self.Up_conv413 = conv_block(filters[3], filters[2])
        self.Up313 = up_conv(filters[2], filters[1])
        self.Up_conv313 = conv_block(filters[2], filters[1])
        self.Up213 = up_conv(filters[1], filters[0])
        self.Up_conv213 = conv_block(filters[1], filters[0])
        self.Up514 = up_conv(2048, 1024)
        self.Up_conv514 = conv_block(2048, 1024)
        self.Up414 = up_conv(1024, 512)
        self.Up_conv414 = conv_block(1024, 512)
        self.Up314 = up_conv(512, 256)
        self.Up_conv314 = conv_block(512, 256)
        self.Up214 = up_conv(256, 128)
        self.Up_conv214 = conv_block(256, 32)

        self.Conv611 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv612 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv613 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv614 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.Last_conv011 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        self.Last_conv012 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        self.Last_conv013 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, art, pv, delay):
        adv = torch.cat((art, pv, delay), dim=1)

        e111 = self.Conv111(art)
        e112 = self.Conv112(delay)
        e113 = self.Conv113(pv)
        e114 = self.Conv114(adv)

        # Apply self-attention to e111 and e112
        e111_attn = self.self_attn1(e111)
        e112_attn = self.self_attn1(e112)

        e114 = torch.cat((e111_attn, e112_attn, e113, e114), dim=1)
        e115 = self.Conv115(e114)

        e211 = self.Maxpool1(e111)
        e212 = self.Maxpool1(e112)
        e213 = self.Maxpool1(e113)
        e214 = self.Maxpool1(e115)

        e211 = self.Conv211(e211)
        e212 = self.Conv212(e212)
        e213 = self.Conv213(e213)
        e214 = self.Conv214(e214)

        # Apply self-attention to e211 and e212
        e211_attn = self.self_attn2(e211)
        e212_attn = self.self_attn2(e212)

        e214 = torch.cat((e211_attn, e212_attn, e213, e214), dim=1)
        e215 = self.Conv215(e214)

        e311 = self.Maxpool2(e211)
        e312 = self.Maxpool2(e212)
        e313 = self.Maxpool2(e213)
        e314 = self.Maxpool2(e215)

        e311 = self.Conv311(e311)
        e312 = self.Conv312(e312)
        e313 = self.Conv313(e313)
        e314 = self.Conv314(e314)

        # Apply self-attention to e311 and e312
        e311_attn = self.self_attn3(e311)
        e312_attn = self.self_attn3(e312)

        e314 = torch.cat((e311_attn, e312_attn, e313, e314), dim=1)
        e315 = self.Conv315(e314)

        e411 = self.Maxpool3(e311)
        e412 = self.Maxpool3(e312)
        e413 = self.Maxpool3(e313)
        e414 = self.Maxpool3(e315)

        e411 = self.Conv411(e411)
        e412 = self.Conv412(e412)
        e413 = self.Conv413(e413)
        e414 = self.Conv414(e414)

        # Apply self-attention to e411 and e412
        e411_attn = self.self_attn4(e411)
        e412_attn = self.self_attn4(e412)

        e414 = torch.cat((e411_attn, e412_attn, e413, e414), dim=1)
        e415 = self.Conv415(e414)

        e511 = self.Maxpool4(e411)
        e512 = self.Maxpool4(e412)
        e513 = self.Maxpool4(e413)
        e514 = self.Maxpool4(e415)

        e511 = self.Conv511(e511)
        e512 = self.Conv512(e512)
        e513 = self.Conv513(e513)
        e514 = self.Conv514(e514)

        # Apply self-attention to e511 and e512
        e511_attn = self.self_attn5(e511)
        e512_attn = self.self_attn5(e512)

        e514 = torch.cat((e511_attn, e512_attn, e513, e514), dim=1)
        e515 = self.Conv515(e514)

        d511 = self.Up511(e511)
        d512 = self.Up512(e512)
        d513 = self.Up513(e513)
        d514 = self.Up514(e515)

        d511 = torch.cat((e411, d511), dim=1)
        d512 = torch.cat((e412, d512), dim=1)
        d513 = torch.cat((e413, d513), dim=1)
        d514 = torch.cat((e415, d514), dim=1)

        d611 = self.Up_conv511(d511)
        d612 = self.Up_conv512(d512)
        d613 = self.Up_conv513(d513)
        d614 = self.Up_conv514(d514)

        d411 = self.Up411(d611)
        d412 = self.Up412(d612)
        d413 = self.Up413(d613)
        d414 = self.Up414(d614)

        d411 = torch.cat((e311, d411), dim=1)
        d412 = torch.cat((e312, d412), dim=1)
        d413 = torch.cat((e313, d413), dim=1)
        d414 = torch.cat((e315, d414), dim=1)

        d411 = self.Up_conv411(d411)
        d412 = self.Up_conv412(d412)
        d413 = self.Up_conv413(d413)
        d414 = self.Up_conv414(d414)

        d311 = self.Up311(d411)
        d312 = self.Up312(d412)
        d313 = self.Up313(d413)
        d314 = self.Up314(d414)

        d311 = torch.cat((e211, d311), dim=1)
        d312 = torch.cat((e212, d312), dim=1)
        d313 = torch.cat((e213, d313), dim=1)
        d314 = torch.cat((e215, d314), dim=1)

        d311 = self.Up_conv311(d311)
        d312 = self.Up_conv312(d312)
        d313 = self.Up_conv313(d313)
        d314 = self.Up_conv314(d314)

        d211 = self.Up211(d311)
        d212 = self.Up212(d312)
        d213 = self.Up213(d313)
        d214 = self.Up214(d314)

        d211 = torch.cat((e111, d211), dim=1)
        d212 = torch.cat((e112, d212), dim=1)
        d213 = torch.cat((e113, d213), dim=1)
        d214 = torch.cat((e115, d214), dim=1)

        d211 = self.Up_conv211(d211)
        d212 = self.Up_conv212(d212)
        d213 = self.Up_conv213(d213)
        d214 = self.Up_conv214(d214)

        out1 = self.Conv611(d211)
        out2 = self.Conv612(d212)
        out3 = self.Conv613(d213)
        out4 = self.Conv614(d214)

        return out4
