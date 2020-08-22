import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url

#########################################################################
## VGG19 pretrained models
#########################################################################
__all__ = [
    'VGG', 'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features[1]
        self.feature_names = features[0]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, extract_feats=[]):
        desired_feats = []
        for i, feat in enumerate(self.features):
            x = feat(x)
            if self.feature_names[i] in extract_feats:
                desired_feats += [x]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, desired_feats

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    layer_names = []
    in_channels = 3
    i = 1
    j = 1
    for v in cfg:
        if v == 'M':
            layer_names += ['max_pool{}'.format(i)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            i += 1
            j = 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layer_names += ['conv{}_{}'.format(i, j), 'batch_norm{}_{}'.format(i, j), 'relu{}_{}'.format(i, j)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                layer_names += ['conv{}_{}'.format(i, j), 'relu{}_{}'.format(i, j)]
            in_channels = v
            j += 1
    return (layer_names, nn.ModuleList(layers))

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

#########################################################################
## Perceptual loss
#########################################################################
class PerceptualLoss():
    def __init__(self, extract_feats=['conv5_4']):
        self.extract_feats = extract_feats
        self.vgg = vgg19(pretrained=True).cuda()
        self.mse = nn.MSELoss()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def __call__(self, learned, real):
        if learned.size(1) == 1:
            learned = learned.repeat(1, 3, 1, 1)
        learned = (learned - self.mean) / self.std
        if learned.size(-1) != 224:
            learned = F.interpolate(learned, mode='bilinear', size=(224, 224), align_corners=False)
        _, learned_feats = self.vgg(learned, extract_feats=self.extract_feats)

        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
        real = (real - self.mean) / self.std
        if real.size(-1) != 224:
            real = F.interpolate(real, mode='bilinear', size=(224, 224), align_corners=False)
        _, real_feats = self.vgg(real, extract_feats=self.extract_feats)

        loss = 0.0
        for i in range(len(real_feats)):
            loss += self.mse(learned_feats[i], real_feats[i].detach())
        return loss

if __name__=="__main__":
    model = vgg19(pretrained=True)
    x = torch.zeros(2, 3, 224, 224)
    y, feats = model(x, extract_feats=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2'])
    print(y.shape, len(feats))
    print(model.feature_names)
