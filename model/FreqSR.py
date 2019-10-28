import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom
from .FastHarleyTransform import *
from .BatchElemMultiplication import *

class FreqSR(nn.Module):
    def __init__(self, shape=(3, 128, 128), scale_factor=2):
        super(FreqSR, self).__init__()
        nc, h, w = shape
        out_shape = (nc, scale_factor * h, scale_factor * w)
        self.upscale = lambda x : F.interpolate(x, scale_factor=(scale_factor, scale_factor),
                                                mode="bilinear", align_corners=False)
        self.fht2d = FHT2D((scale_factor * h, scale_factor * w))
        self.block1 = FreqSRBlock(out_shape, nc * 5)
        self.block2 = FreqSRBlock(out_shape, nc * 5)
        self.block3 = FreqSRBlock(out_shape, nc * 5)
        self.block4 = FreqSRBlock(out_shape, nc * 5)
        self.conv1x1 = nn.Conv2d(nc * 4, nc, kernel_size=1)

    def forward(self, x):
        # x_up = self.upscale(x)
        x_up = x
        x_freq = self.fht2d(x_up)
        x_layer1 = self.block1(x_freq)
        x_layer2 = self.block2(x_layer1)
        x_layer3 = self.block3(x_layer2)
        x_layer4 = self.block4(x_layer3)
        x_layers = torch.cat([x_layer1, x_layer2, x_layer3, x_layer4], dim=1)
        # x_layers = x_layer1
        x_proj = self.conv1x1(x_layers)
        x_spac = self.fht2d(x_proj, inverse=True)
        return x_up + x_spac

class FreqSRBlock(nn.Module):
    def __init__(self, shape, oc):
        super(FreqSRBlock, self).__init__()
        nc, h, w = shape
        block = [BatchElemMultiplication(shape, oc),
                 nn.Conv2d(oc, nc, kernel_size=5, padding=2)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# -------------------- TEST --------------------- #
class FreqSR2(nn.Module):
    def __init__(self, shape=(3, 128, 128), scale_factor=2):
        super(FreqSR2, self).__init__()
        nc, h, w = shape
        out_shape = (nc, scale_factor * h, scale_factor * w)
        self.fht2d = FHT2D((scale_factor * h, scale_factor * w))
        self.block1 = FreqSRBlock2(out_shape, nc * 5)
        self.block2 = FreqSRBlock2(out_shape, nc * 5)
        self.block3 = FreqSRBlock2(out_shape, nc * 5)
        self.block4 = FreqSRBlock2(out_shape, nc * 5)
        self.conv1x1 = nn.Conv2d(nc * 4, nc, kernel_size=1)

    def forward(self, x):
        # x_up = self.upscale(x)
        x_up = x
        x_freq = self.fht2d(x_up)
        x_layer1 = self.block1(x_freq)
        x_layer2 = self.block2(x_layer1)
        x_layer3 = self.block3(x_layer2)
        x_layer4 = self.block4(x_layer3)
        x_layers = torch.cat([x_layer1, x_layer2, x_layer3, x_layer4], dim=1)
        # x_layers = x_layer1
        x_proj = self.conv1x1(x_layers)
        x_spac = self.fht2d(x_proj, inverse=True)
        return x_up + x_spac

class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class Mult(nn.Module):
    def __init__(self, shape):
        super(Mult, self).__init__()
        nc, h, w = shape

        self.weights = nn.Parameter(data=torch.Tensor(1, nc, h, w))
        nn.init.xavier_uniform_(self.weights.data)
        # nn.init.uniform_(self.weights.data, -0.5, 0.5)

    def forward(self, x):
        return x * self.weights

class FreqSRBlock2(nn.Module):
    def __init__(self, shape, oc):
        super(FreqSRBlock2, self).__init__()
        nc, h, w = shape

        self.n = oc // nc
        self.mults = ListModule(self, 'mults_')
        self.convs = ListModule(self, 'convs_')
        for i in range(self.n):
            self.mults.append(Mult(shape))
            self.convs.append(nn.Conv2d(nc, nc, kernel_size=5, padding=2))

    def forward(self, x):
        outputs = 0.0
        for i in range(self.n):
            output = self.mults[i](x)
            outputs += self.convs[i](output)

        return outputs


if __name__=="__main__":
    model = FreqSR2((3, 128, 128), scale_factor=2)
    x = torch.zeros((4, 3, 256, 256))
    y = model(x)
    print(y.shape)
    for name, param in model.named_parameters():
        print(name)