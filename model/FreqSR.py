import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLayer(nn.Module):
    def __init__(self, shape, bias=True):
        super(WeightedLayer, self).__init__()
        nc, h, w = shape

        self.weights = nn.Parameter(data=torch.Tensor(1, nc, h, w))
        # nn.init.xavier_uniform_(self.weights.data)
        nn.init.uniform_(self.weights.data, -1, 1)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(1, nc, h, w))
            # nn.init.xavier_uniform_(self.bias.data)
            nn.init.uniform_(self.bias.data, -1, 1)

    def forward(self, x):
        x = (self.weights + self.weights.permute(0, 1, 3, 2)) / 2 * x
        if self.bias is not None:
            x = x + (self.bias + self.bias.permute(0, 1, 3, 2)) / 2
        return x

class FreqSRBlock(nn.Module):
    def __init__(self, shape, oc=16, kernel_size=7):
        super(FreqSRBlock, self).__init__()
        nc, h, w = shape

        self.weightlayer = WeightedLayer(shape)
        convs = [nn.Conv2d(nc, oc, kernel_size=kernel_size, padding=3, bias=True),
                 nn.Conv2d(oc, oc, kernel_size=kernel_size, padding=3, bias=True),
                 nn.Conv2d(oc, nc, kernel_size=kernel_size, padding=3, bias=True)]
        self.convs = nn.ModuleList(convs)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.weightlayer(x)
        for conv in self.convs:
            x = conv(x)
            x = self.tanh(x)
        return x

class FreqSR(nn.Module):
    """
    This expects a frequency matrix,
    Input is expected to already be scaled up to the desired factor using bicubic

    Paper expects only Y channel from YCbCr, passing in grayscale is okay
    """
    def __init__(self, shape=(1, 128, 128), nlayers=3):
        super(FreqSR, self).__init__()
        nc, h, w = shape
        layers = []
        for i in range(nlayers):
            layers.append(FreqSRBlock(shape))
        self.layers = nn.ModuleList(layers)
        self.conv1x1 = nn.Conv2d(nlayers, nc, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.conv1x1(x)
        return self.tanh(x + residual)


if __name__=="__main__":
    model = FreqSR((1, 256, 256))
    x = torch.zeros((4, 1, 256, 256))
    y = model(x)
    print(y.shape)
    for name, param in model.named_parameters():
        print(name)
