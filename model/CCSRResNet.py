import torch
import torch.nn as nn
import math

from .CoordConv import *

"""
Code in this file is taken from https://github.com/twtygqyy/pytorch-SRResNet
"""

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = CoordConv(in_channels=64, out_channels=64, with_r=True, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = CoordConv(in_channels=64, out_channels=64, with_r=True, kernel_size=3, stride=1, padding=1, bias=True)
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

        self.conv_input = CoordConv(in_channels=nc, out_channels=64, with_r=True, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(Residual_Block, 16)

        self.conv_mid = CoordConv(in_channels=64, out_channels=64, with_r=True, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale = None
        if upscale == 2:
            self.upscale = nn.Sequential(
                CoordConv(in_channels=64, out_channels=256, with_r=True, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif upscale == 4:
            self.upscale = nn.Sequential(
                CoordConv(in_channels=64, out_channels=256, with_r=True, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                CoordConv(in_channels=64, out_channels=256, with_r=True, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = CoordConv(in_channels=64, out_channels=nc, with_r=True, kernel_size=9, stride=1, padding=4, bias=True)

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
        if self.upscale:
            out = self.upscale(out)
        out = self.conv_output(out)
        return torch.sigmoid(out)

class Discriminator(nn.Module):
    def __init__(self, nc=3, nlayers=4):
        super(Discriminator, self).__init__()

        self.features = []

        # input is (nc) x W x H
        self.features += [nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.LeakyReLU(0.2, inplace=True)]
        # state size. (64) x W x H
        self.features += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
                          nn.BatchNorm2d(64),
                          nn.LeakyReLU(0.2, inplace=True)]

        for i in range(nlayers - 1):
            nc = min((2 ** i) * 64 , 512)
            oc = min((2 ** (i + 1)) * 64, 512)
            self.features += [nn.Conv2d(in_channels=nc, out_channels=oc, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.BatchNorm2d(oc),
                              nn.LeakyReLU(0.2, inplace=True),
                              nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=4, stride=2, padding=1, bias=True),
                              nn.BatchNorm2d(oc),
                              nn.LeakyReLU(0.2, inplace=True)]

        self.features = nn.Sequential(*self.features)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        # state size. (512) x 8 x 8
        x = x.view(x.size(0), -1)
        # state size. (512 x 8 x 8)
        x = self.fc1(x)
        # state size. (1024)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        # out = self.sigmoid(out)
        return x.view(-1, 1).squeeze(1)


if __name__=="__main__":
    model = SRResNet(nc=1, upscale=2)
    x = torch.zeros((4, 1, 64, 64))
    y = model(x)
    print(y.shape)

    discriminator = Discriminator(nc=1, nlayers=4)
    z = discriminator(y)
    print(z.shape)

    # from torchsummary import summary
    # summary(model, (1, 64, 64))
