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
        x_up = self.upscale(x)
        x_freq = self.fht2d(x_up, forward=True)
        x_layer1 = self.block1(x_freq)
        x_layer2 = self.block2(x_layer1)
        x_layer3 = self.block3(x_layer2)
        x_layer4 = self.block4(x_layer3)
        x_layers = torch.cat([x_layer1, x_layer2, x_layer3, x_layer4], dim=1)
        x_proj = self.conv1x1(x_layers)
        x_spac = self.fht2d(x_proj, forward=False)
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


if __name__=="__main__":
    model = FreqSR((3, 128, 128), scale_factor=2)
    x = torch.zeros((4, 3, 128, 128))
    y = model(x)
    print(y.shape)