import os
import torch
import numpy as np
import scipy.ndimage as ndimage

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from skimage.transform import radon, iradon

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


class DeepLesionDataset(Dataset):
    def __init__(self, root, image_shape=(256, 256), add_noise=True, preprocessed=False, crop_size=None):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.image_shape = np.asarray(image_shape)
        self.add_noise = add_noise
        self.preprocessed = preprocessed
        self.crop_size = crop_size
        # directory structure
        # STUDY ID
        #     |
        #     |_ _slice number
        self.studies = [d for d in os.listdir(root) if '.DS' not in d]
        self.samples = []
        for d in self.studies:
            self.samples += [os.path.join(d, f) for f in os.listdir(os.path.join(root, d)) if '.png' in f]
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

    def ct_noise(self, image, alpha=1, beta=0.333):
        ct = np.asarray(image)
        h, w = ct.shape
        theta = np.linspace(0., 180., max(ct.shape), endpoint=False)
        sino = radon(ct, theta=theta, circle=True)
        mask = np.random.binomial(1, 0.98, sino.shape)
        nsino = np.random.poisson(sino * mask / 2)
        noise = iradon(nsino, theta=theta, circle=True)
        noise = noise - noise[0, 0]
        noise = (255 * ((noise - noise.min()) / (noise.max() - noise.min())))
        noise = alpha * ct + beta * noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = (2 * noise + np.random.poisson((noise * 255).astype(np.uint8)) / 255.0) / 3
        return Image.fromarray(noise)

    def __getitem__(self, idx):
        if self.preprocessed:
            imageHR = Image.open(self.at(idx))
            imageLR = Image.open(self.at(idx).replace('miniStudies', 'noiseStudies'))
        else:
            image = np.asarray(Image.open(self.at(idx)), dtype=np.int32) - 32768
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
            image = Image.fromarray(image)
            if self.crop_size:
                crop_size = np.asarray(self.crop_size)
                imageHR = self.random_crop(image)
                imageLR = imageHR.resize(crop_size[::-1] // self.scale_factor, resample=Image.LANCZOS)
            else:
                imageHR = image.resize(self.image_shape[::-1], resample=Image.LANCZOS)
                imageLR = imageHR

            if self.add_noise:
                imageLR = self.ct_noise(imageLR)

        return (self.totensor(imageLR), self.totensor(imageHR))

    def at(self, idx):
        return os.path.join(self.root, self.samples[idx])


if __name__=="__main__":
    # which = 'Noisy'
    # which = 'CXR8'
    which = 'DeepLesion'
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
    elif which == 'DeepLesion':
        # datapath = os.path.expanduser("~/Downloads/DeepLesion2/Images_png/")
        # dataset = DeepLesionDataset(datapath, add_noise=True)
        datapath = os.path.expanduser("~/Downloads/DeepLesion2/miniStudies/")
        dataset = DeepLesionDataset(datapath, preprocessed=True)
        print(len(dataset))
        print(dataset.at(0))

        import time
        start = time.time()
        imageLR, imageHR = dataset[0]
        total = time.time() - start
        print(total, 'seconds')
        print(imageHR.shape)
        imageLR = np.asarray(imageLR[0]) #.transpose(1, 2, 0)
        imageHR = np.asarray(imageHR[0]) #.transpose(1, 2, 0)
        print(imageHR.shape)
        print(imageLR.max(), imageLR.min())
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        axes[0].imshow(imageLR, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Noise')
        axes[1].imshow(imageHR, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('CT')
        fig.tight_layout()
        plt.show()
