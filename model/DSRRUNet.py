import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.block = []
        for i in range(3):
            ic = (64, channels)[i == 0]
            oc = (64, channels)[i == 2]
            ks = (1, 3)[i == 1]
            pd = (0, 1)[i == 1]
            self.block += [nn.Conv2d(ic, oc, kernel_size=ks, padding=pd, bias=True),
                           nn.InstanceNorm2d(oc, affine=True),
                           nn.LeakyReLU(0.2, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        identity_data = x
        output = self.block(x) + identity_data
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = []
        channels = in_channels
        for _ in range(3):
            self.block += [nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True),
                           nn.BatchNorm2d(out_channels, affine=True),
                           nn.LeakyReLU(inplace=True)]
            channels = out_channels

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

class PixelShuffleBlock(nn.Module):
    def __init__(self, channels):
        super(PixelShuffleBlock, self).__init__()

        self.block = [nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1, bias=True),
                      nn.PixelShuffle(2),
                      nn.LeakyReLU(0.2, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

class DSRRUNet(nn.Module):
    def __init__(self, nc=1, upscale=4, residual_layers=8):
        super(DSRRUNet, self).__init__()
        # check input parameters
        assert residual_layers // 4 > 0, "residual layers passed in is not large enough"

        # U-Net porition
        ## encoder
        channels = [nc, 64, 128, 256]
        self.encoder = []
        for i in range(3):
            self.encoder += [ConvBlock(channels[i], channels[i + 1])]
        self.encoder = nn.ModuleList(self.encoder)
        ## down sample
        self.down = nn.MaxPool2d(2)
        ## bottle neck
        self.bottleneck = ConvBlock(256, 256)
        ## decoder
        channels = [512, 256, 128, 64]
        skip_nc  = [  0, 128,  64,  0]
        self.decoder = []
        for i in range(3):
            self.decoder += [ConvBlock(channels[i] + skip_nc[i], channels[i + 1])]
        self.decoder = nn.ModuleList(self.decoder)
        ## upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ResNet blocks
        self.residualblocks = []
        for i in range(3):
            resblock = []
            for _ in range(residual_layers):
                resblock += [ResBlock(channels[-i - 1])]
            resblock = nn.Sequential(*resblock)
            self.residualblocks += [resblock]
            residual_layers //= 2
        self.residualblocks = nn.ModuleList(self.residualblocks)

        self.midconv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                     nn.InstanceNorm2d(64, affine=True))

        self.pixelshuffle = []
        for _ in range(upscale // 2):
            self.pixelshuffle += [PixelShuffleBlock(64)]
        self.pixelshuffle = nn.Sequential(*self.pixelshuffle)

        self.finalconv = nn.Conv2d(64, out_channels=nc, kernel_size=9, padding=4, bias=True)

    def forward(self, x):
        # store outputs
        skip = []
        identity = None

        # encoder
        for i, (elayer, rlayer) in enumerate(zip(self.encoder, self.residualblocks)):
            x = elayer(x)
            identity = (identity, x)[i == 0]
            skip += [rlayer(x)]
            x = self.down(x)

        # bottle neck layer
        x = self.bottleneck(x)

        # decoder
        for layer in self.decoder:
            x = self.up(x)
            x = torch.cat([x, skip.pop()], dim=1)
            x = layer(x)

        # sr resnet
        x = self.midconv(x) + identity
        x = self.pixelshuffle(x)
        x = self.finalconv(x)

        return torch.sigmoid(x)


if __name__=="__main__":
    model = DSRRUNet(upscale=1, residual_layers=8)
    x = torch.zeros((4, 1, 64, 64))
    y = model(x)
    print(y.shape)

    # torch.save({'state_dict' : model.state_dict()}, 'dsrnet.pth')
    # torch.save(model.state_dict(), 'dsrnet.pth')
    # torch.save(model, 'dsrnet.pth')

    # from torchsummary import summary
    # summary(model, (1, 64, 64))