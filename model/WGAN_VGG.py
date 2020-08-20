import torch
import torch.nn as nn
import math

"""
Code in this file is taken from https://github.com/twtygqyy/pytorch-SRResNet
"""

class WGAN_VGG(nn.Module):
    def __init__(self, nc=1):
        super(WGAN_VGG, self).__init__()

        layers = []
        for i in range(8):
            nc = 32 if i > 0 else nc
            oc = 32 if i != 7 else 1
            layers += [nn.Conv2d(nc, oc, kernel_size=3, stride=1, padding=1, bias=True),
                       nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return torch.sigmoid(out)

class Discriminator(nn.Module):
    def __init__(self, nc=1, nlayers=4):
        super(Discriminator, self).__init__()

        features = []
        for i in range(3):
            nc = oc if i > 0 else nc
            oc = oc * 2 if i > 0 else 64
            features += [nn.Conv2d(nc, oc, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(oc, oc, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True)]

        self.features = nn.Sequential(*features)

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)

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
    model = WGAN_VGG(nc=1)
    x = torch.zeros((4, 1, 64, 64))
    y = model(x)
    print(y.shape)

    discriminator = Discriminator(nc=1)
    z = discriminator(y)
    print(z.shape)

    # from torchsummary import summary
    # summary(model, (1, 64, 64))
