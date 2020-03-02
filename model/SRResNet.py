import torch
import torch.nn as nn
import math

"""
Code in this file is taken from https://github.com/twtygqyy/pytorch-SRResNet
"""

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output

class SRResNet(nn.Module):
    def __init__(self, nc=3, upscale=4):
        super(SRResNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        if upscale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif upscale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=nc, kernel_size=9, stride=1, padding=4, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale(out)
        out = self.conv_output(out)
        return torch.sigmoid(out)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = (None, nn.BatchNorm2d(out_channels))[int(batchnorm)]
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, nc=3):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # input is 1 x 128 x 128
            ConvBlock(in_channels=nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False),

            # state size. 64 x 128 x 128
            ConvBlock(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True, batchnorm=True),

            # state size. 64 x 128 x 128
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True),

            # state size. 128 x 64 x 64
            ConvBlock(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True, batchnorm=True),

            # state size. 128 x 64 x 64
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True),

            # state size. 256 x 32 x 32
            ConvBlock(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True, batchnorm=True),

            # state size. 256 x 32 x 32
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True),

            # state size. 512 x 16 x 16
            ConvBlock(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True, batchnorm=True),

            # state size. 512 x 8 x 8
            ConvBlock(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True, batchnorm=True)
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        # 128 x 128 -> 512 * 4 * 4 // Added another layer to reduce size more
        # 96  x 96  -> 512 * 6 * 6
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, features=False):
        x = self.features(x)

        # state size. (512) x 4 x 4
        x = x.view(x.size(0), -1)

        # state size. (512 x 4 x 4)
        x = self.fc1(x)
        if features:
            feats = x

        # state size. (1024)
        x = self.LeakyReLU(x)

        x = self.fc2(x)
        # out = self.sigmoid(out)
        if features:
            return x.view(-1, 1).squeeze(1), feats
        return x.view(-1, 1).squeeze(1)


if __name__=="__main__":
    model = SRResNet(nc=1, upscale=2)
    x = torch.zeros((4, 1, 64, 64))
    y = model(x)
    print(y.shape)

    discriminator = Discriminator(nc=1)
    z = discriminator(y)
    print(z.shape)
