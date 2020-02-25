import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# this is left over from natural images
# from .image_utils import *


class CXR8Dataset(Dataset):
    def __init__(self, root, image_shape=(256, 256), scale_factor=4, color='grayscale', upsample='bicubic'):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.image_shape = np.asarray(image_shape)
        self.scale_factor = scale_factor
        self.color = color
        self.upsample = upsample
        self.samples = [sample for sample in os.listdir(root) if '.png' in sample]
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


class NoisyXrayDataset(Dataset):
    def __init__(self, root, train=True):
        # root should be the path to xray_images
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.dir = ['test_images_64x64', 'train_images_64x64'][int(train)]
        self.samples = [sample for sample in os.listdir(os.path.join(root, self.dir)) if '.png' in sample]
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pathLR = self.at(idx)
        pathHR = pathLR.replace('64x64', '128x128')
        imageLR = Image.open(pathLR).convert('L')
        imageHR = Image.open(pathHR).convert('L')

        return (self.totensor(imageLR), self.totensor(imageHR))

    def at(self, idx):
        return os.path.join(self.root, self.dir, self.samples[idx])


if __name__=="__main__":
    which = 'Noisy'
    if which == 'CX48':
        datapath = os.path.expanduser("~/Projects/datasets/CXR8/images/")
        dataset = CXR8Dataset(datapath, color="grayscale")
        print(len(dataset))
        print(dataset.at(0))

        imageLR, imageHR = dataset[0]
        print(imageHR.shape)
        imageLR = np.asarray(imageLR[0]) #.transpose(1, 2, 0)
        imageHR = np.asarray(imageHR[0]) #.transpose(1, 2, 0)
        print(imageHR.shape)
        import matplotlib.pyplot as plt
        plt.imshow(imageHR, cmap="gray")
        plt.show()
    elif which == 'Noisy':
        datapath = os.path.expanduser("~/Projects/datasets/xray_images/")
        dataset = NoisyXrayDataset(datapath, train=False)
        print(len(dataset))
        print(dataset.at(0))

        imageLR, imageHR = dataset[0]
        print(imageHR.shape)
        imageLR = np.asarray(imageLR[0]) #.transpose(1, 2, 0)
        imageHR = np.asarray(imageHR[0]) #.transpose(1, 2, 0)
        print(imageHR.shape)
        import matplotlib.pyplot as plt
        plt.imshow(imageHR, cmap="gray")
        plt.show()
