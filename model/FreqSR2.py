import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqSRBlock2(nn.Module):
    def __init__(self, nc=64, oc=64, kernel_size=3, padding=1, nlayers=2):
        super(FreqSRBlock2, self).__init__()

        layers = []
        for layer in range(nlayers):
            layers.append(nn.Sequential(nn.Conv2d(nc, oc, kernel_size=kernel_size, padding=padding, bias=True),
                                        nn.BatchNorm2d(oc),
                                        nn.Tanh(),
                                        nn.Conv2d(oc, oc, kernel_size=kernel_size, padding=padding, bias=True),
                                        nn.BatchNorm2d(oc),
                                        nn.Tanh()))
            nc = oc

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x) + x
        return x

class FreqSR2(nn.Module):
    """
    This expects a frequency matrix,
    Input is expected to already be scaled up to the desired factor using bicubic
    """
    def __init__(self, nc=3, expand=64, nlayers=3):
        super(FreqSR2, self).__init__()
        self.convFirst = nn.Sequential(nn.Conv2d(nc, expand, kernel_size=1, bias=True),
                                       nn.BatchNorm2d(expand),
                                       nn.Tanh())
        self.convLast = nn.Sequential(nn.Conv2d(expand, nc, kernel_size=1, bias=True),
                                      nn.Tanh())

        layers = []

        for layer in range(nlayers):
            layers.append(FreqSRBlock2(expand, expand, kernel_size=3, padding=1, nlayers=2))
            layers.append(nn.Sequential(nn.Conv2d(expand * 2, expand, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(expand),
                                        nn.Tanh()))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.convFirst(x)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x = torch.cat((x, layer(x)), dim=1)
            else:
                x = layer(x)

        return self.convLast(x)


if __name__=="__main__":
    model = FreqSR2(1, 64, 3)
    # model = FreqSRBlock2(1, 32)
    x = torch.zeros((4, 1, 180, 240))
    y = model(x)
    print(y.shape)
    # for name, param in model.named_parameters():
    #     print(name)
    import torchsummary
    torchsummary.summary(model, (1, 180, 240))
