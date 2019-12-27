import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .image_utils import *


class VOC2012(Dataset):
    def __init__(self, root, image_shape=(256, 256), scale_factor=2, color='rgb', upsample='bicubic'):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.image_shape = np.asarray(image_shape)
        self.scale_factor = scale_factor
        self.color = color
        self.upsample = upsample
        self.samples = os.listdir(root)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Use gray scale for now
        image = Image.open(self.at(idx))
        imageHR = image.resize(self.image_shape[::-1], resample=Image.LANCZOS)
        imageLR = image.resize(self.image_shape[::-1] // self.scale_factor, resample=Image.LANCZOS)
        if self.upsample == 'bicubic':
            imageLR = imageLR.resize(self.image_shape[::-1], resample=Image.BICUBIC)
        elif self.upsample == 'bilinear':
            imageLR = imageLR.resize(self.image_shape[::-1], resample=Image.BILINEAR)

        if self.color == 'ycbcr':
            yLR, ycbcrLR = preprocess(imageLR)
            yHR, ycbcrHR = preprocess(imageHR)
            return (yLR, yHR)
        elif self.color == 'grayscale':
            imageLR = imageLR.convert('L')
            imageHR = imageHR.convert('L')

        return (self.totensor(imageLR), self.totensor(imageHR))

    def at(self, idx):
        return os.path.join(self.root, self.samples[idx])


if __name__=="__main__":
    datapath = os.path.expanduser("~/Downloads/VOCdevkit/VOC2012/JPEGImages/")
    dataset = VOC2012(datapath)
    print(len(dataset))
    print(dataset.at(0))

    imageLR, imageHR = dataset[0]
    imageLR = np.asarray(imageLR).transpose(1, 2, 0)
    imageHR = np.asarray(imageHR).transpose(1, 2, 0)
    import matplotlib.pyplot as plt
    plt.imshow(imageHR)
    plt.show()
