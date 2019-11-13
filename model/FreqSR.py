import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLayer(nn.Module):
    def __init__(self, shape, bias=True):
        super(WeightedLayer, self).__init__()
        nc, h, w = shape

        self.nc = nc
        self.h = h // 2
        self.w = w // 2
        self.weights = nn.Parameter(data=torch.Tensor(1, nc, self.h, self.w), requires_grad=True)
        nn.init.constant_(self.weights.data, 0.1)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(1, nc, self.h, self.w), requires_grad=True)
            nn.init.constant_(self.bias.data, 0.0)

    def _mult(self, x, weights):
        # [W_2 W_1]
        # [W_3 W_4]
        # W_1 = fliplr(W_2) = fliplr(flipud(W_3)) = flipud(W_4)
        W = torch.zeros(1, self.nc, 2 * self.h, 2 * self.w).to(x.device)
        W[:,:,:self.h,self.w:] = weights
        W[:,:,:self.h,:self.w] = torch.flip(weights, [3])
        W[:,:,self.h:,self.w:] = torch.flip(weights, [2])
        W[:,:,self.h:,:self.w] = torch.flip(weights, [2,3])
        x = W * x
        # multiply the top right corner
        # x[:,:,:self.h,self.w:] = weights * x[:,:,:self.h,self.w:]
        # # multiply the top left corner
        # x[:,:,:self.h,:self.w] = torch.flip(weights, [3]) * x[:,:,:self.h,:self.w]
        # # mutiply the bottom right corner
        # x[:,:,self.h:,self.w:] = torch.flip(weights, [2]) * x[:,:,self.h:,self.w:]
        # # mutiply the bottom left corner
        # x[:,:,self.h:,:self.w] = torch.flip(weights, [2,3]) * x[:,:,self.h:,:self.w]
        return x

    def _add(self, x, weights):
        # [W_2 W_1]
        # [W_3 W_4]
        # W_1 = fliplr(W_2) = fliplr(flipud(W_3)) = flipud(W_4)
        W = torch.zeros(1, self.nc, 2 * self.h, 2 * self.w).to(x.device)
        W[:,:,:self.h,self.w:] = weights
        W[:,:,:self.h,:self.w] = torch.flip(weights, [3])
        W[:,:,self.h:,self.w:] = torch.flip(weights, [2])
        W[:,:,self.h:,:self.w] = torch.flip(weights, [2,3])
        x = W + x
        # multiply the top right corner
        # x[:,:,:self.h,self.w:] = weights + x[:,:,:self.h,self.w:]
        # # multiply the top left corner
        # x[:,:,:self.h,:self.w] = torch.flip(weights, [3]) + x[:,:,:self.h,:self.w]
        # # mutiply the bottom right corner
        # x[:,:,self.h:,self.w:] = torch.flip(weights, [2]) + x[:,:,self.h:,self.w:]
        # # mutiply the bottom left corner
        # x[:,:,self.h:,:self.w] = torch.flip(weights, [2,3]) + x[:,:,self.h:,:self.w]
        return x

    def forward(self, x):
        x = self._mult(x, self.weights)
        if self.bias is not None:
            x = self._add(x, self.bias)
        return x

class FreqSRBlock(nn.Module):
    def __init__(self, shape, oc=16, kernel_size=7, padding=3):
        super(FreqSRBlock, self).__init__()
        nc, h, w = shape

        self.weightlayer = WeightedLayer(shape)
        convs = [nn.Conv2d(nc, oc, kernel_size=kernel_size, padding=padding, bias=True),
                 nn.Conv2d(oc, oc, kernel_size=kernel_size, padding=padding, bias=True),
                 nn.Conv2d(oc, nc, kernel_size=kernel_size, padding=padding, bias=True)]
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
    def __init__(self, shape=(1, 128, 128), nlayers=5):
        super(FreqSR, self).__init__()
        nc, h, w = shape
        layers = []
        for i in range(nlayers):
            layers.append(FreqSRBlock(shape))
        self.layers = nn.ModuleList(layers)
        # self.conv1x1 = nn.Conv2d(nlayers, nc, kernel_size=1)
        self.tanh = nn.Tanh()
        self.weightlayer = WeightedLayer(shape)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        # x = torch.cat(outputs, dim=1)
        # x = self.conv1x1(x)
        x = torch.cat(outputs, dim=1)
        x = torch.sum(x, dim=1).unsqueeze(1)
        x = self.weightlayer(x)
        return self.tanh(x)


if __name__=="__main__":
    model = FreqSR((1, 180, 240))
    x = torch.zeros((4, 1, 180, 240))
    y = model(x)
    print(y.shape)
    # for name, param in model.named_parameters():
    #     print(name)
    import torchsummary
    torchsummary.summary(model, (1, 180, 240))
