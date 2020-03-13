import os
import torch
import numpy as np
import scipy.ndimage as ndimage

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# this is left over from natural images
# from .image_utils import *


class CXR8Dataset(Dataset):
    def __init__(self, root, image_shape=(256, 256), scale_factor=4, add_noise=True, crop_size=None):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.image_shape = np.asarray(image_shape)
        self.scale_factor = scale_factor
        self.add_noise = add_noise
        self.crop_size = crop_size
        self.samples = [sample for sample in os.listdir(root) if ('.png' in sample and '._' not in sample)]
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def random_crop(self, image):
        image = np.asarray(image)
        h, w = image.shape
        c_h, c_w = self.crop_size

        x_start = np.random.randint(0, h - c_h)
        y_start = np.random.randint(0, w - c_w)

        image = image[x_start:x_start + c_h, y_start:y_start + c_w]
        return Image.fromarray(image)

    def __getitem__(self, idx):
        image = Image.open(self.at(idx)).convert('L')
        if self.crop_size:
            crop_size = np.asarray(self.crop_size)
            imageHR = self.random_crop(image)
            imageLR = imageHR.resize(crop_size[::-1] // self.scale_factor, resample=Image.LANCZOS)
        else:
            imageHR = image.resize(self.image_shape[::-1], resample=Image.LANCZOS)
            imageLR = image.resize(self.image_shape[::-1] // self.scale_factor, resample=Image.LANCZOS)

        if self.add_noise:
            imageLR = np.asarray(imageLR)
            nrows, ncols = imageLR.shape
            imageLR = np.random.poisson(imageLR)
            if self.scale_factor == 1:
                imageLR = ndimage.gaussian_filter(imageLR, sigma=1.0)
            imageLR = np.clip(imageLR, 0, 255).astype(np.uint8)
            imageLR = Image.fromarray(imageLR)

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
    # which = 'Noisy'
    which = 'CXR8'
    if which == 'CXR8':
        datapath = os.path.expanduser("~/Projects/datasets/miniCXR8/images/")
        dataset = CXR8Dataset(datapath, scale_factor=1, add_noise=True, crop_size=(96,96))
        print(len(dataset))
        print(dataset.at(0))

        imageLR, imageHR = dataset[0]
        print(imageHR.shape)
        imageLR = np.asarray(imageLR[0]) #.transpose(1, 2, 0)
        imageHR = np.asarray(imageHR[0]) #.transpose(1, 2, 0)
        print(imageHR.shape)
        print(imageLR.max(), imageLR.min())
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(5, 3))
        axes[0].imshow(imageLR, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Noise')
        axes[1].imshow(imageHR, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Xray')
        fig.tight_layout()
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
