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


class DeepLesionDataset(Dataset):
    def __init__(self, root, crop_size=None):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
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

    def random_crop(self, image, x_start=None, y_start=None):
        image = np.asarray(image)
        h, w = image.shape
        c_h, c_w = self.crop_size

        x_start = np.random.randint(0, h - c_h) if x_start == None else x_start
        y_start = np.random.randint(0, w - c_w) if y_start == None else y_start

        image = image[x_start:x_start + c_h, y_start:y_start + c_w]
        return Image.fromarray(image), x_start, y_start

    def __getitem__(self, idx):
        imageHR = Image.open(self.at(idx)).convert('L')
        imageLR = Image.open(self.at(idx).replace('miniStudies', 'noiseStudies')).convert('L')
        if self.crop_size:
            crop_size = np.asarray(self.crop_size)
            imageHR, x, y = self.random_crop(imageHR)
            imageLR, _, _ = self.random_crop(imageLR, x, y)

        return (self.totensor(imageLR), self.totensor(imageHR))

    def at(self, idx):
        return os.path.join(self.root, self.samples[idx])
