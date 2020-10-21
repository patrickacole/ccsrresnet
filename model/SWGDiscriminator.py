import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

            # conv to reduce size
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class SWGDiscriminator(nn.Module):
    def __init__(self, shape=(1, 128, 128), nlayers=5):
        super(SWGDiscriminator, self).__init__()

        channels = [shape[0], 64, 128, 256, 512]

        features = []
        for i in range(nlayers):
            nc = min(i, len(channels) - 1)
            oc = min(i + 1, len(channels) - 1)
            features += [ConvBlock(channels[nc], channels[oc])]

        self.features = nn.Sequential(*features)
        self.fdim = channels[oc] * (shape[1] // (2 ** nlayers)) ** 2
        self.fc = nn.Linear(self.fdim, 1)

    def forward(self, x, features=False):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        labels = self.fc(feats)
        labels = labels.view(-1, 1).squeeze(1)
        if features:
            return labels, feats
        return labels


if __name__=="__main__":
    modelD = SWGDiscriminator(shape=(1, 128, 128), nlayers=5)
    x = torch.zeros((4, 1, 128, 128))
    labels = modelD(x)
    print(labels.shape)
    labels, feats = modelD(x, features=True)
    print(feats.shape)