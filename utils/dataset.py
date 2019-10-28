import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class VOC2012(Dataset):
    def __init__(self, root, image_shape=(256, 256), scale_factor=2, grayscale=False):
        if not os.path.exists(root):
            raise OSError("{}, path does not exist..".format(root))

        self.root = root
        self.image_shape = np.asarray(image_shape)
        self.scale_factor = scale_factor
        self.grayscale = grayscale
        self.samples = os.listdir(root)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Use gray scale for now
        image = Image.open(self.at(idx))
        if self.grayscale:
            image = image.convert('L')
        imageHR = image.resize(self.image_shape[::-1], resample=Image.LANCZOS)
        imageLR = image.resize(self.image_shape[::-1] // self.scale_factor, resample=Image.LANCZOS)
        imageLR = imageLR.resize(self.image_shape[::-1], resample=Image.BICUBIC)

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
