import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import compare_psnr

# custom imports
from utils.dataset import *
from utils.checkpoints import *
from utils.hartleytransform import *
from model.FreqSR import *

# global variables
M, N = (256, 256)
device = None
rgb = False
C = (1, 3)[int(rgb)]


class PIL_to_tensor():
    def __init__(self):
        self.totensor = ToTensor()

    def __call__(self, image):
        return self.totensor(image)


model = FreqSR(shape=(C, M, N))
checkpoint = torch.load(f'checkpoints/{C}x{M}x{N}/best.pth', map_location='cpu')
# Model was trained with nn.DataParallel so need to get rid of `module.` in every key
state_dict = {}
for key in checkpoint['state_dict'].keys():
    state_dict[key.split('module.')[1]] = checkpoint['state_dict'][key]
model.load_state_dict(state_dict)

data_path = "dataset/Set14/image_SRF_2/"
image_paths = os.listdir(data_path)
image_paths = [path for path in image_paths if '_HR' in path]
image_paths = sorted(image_paths)

avg_l_psnr = 0.0
avg_b_psnr = 0.0
max_l_psnr = 0.0
max_b_psnr = 0.0
for path in image_paths:
    hr = Image.open(os.path.join(data_path, path))
    if rgb:
        hr = hr.convert('RGB')
    else:
        hr = hr.convert('L')
    if (hr.size != (N, M)):
        hr = hr.resize((N, M), resample=Image.LANCZOS)
    lr = hr.resize((N // 2, M // 2), resample=Image.LANCZOS)
    bicubic_hr = lr.resize((N, M), resample=Image.BICUBIC)

    convert = PIL_to_tensor()
    fht2d = FHT2D((M,N))
    hr = convert(hr)
    lr = convert(lr)
    bicubic_hr = convert(bicubic_hr)

    freq_lr = fht2d(bicubic_hr.unsqueeze(0))
    learned_hr = fht2d(model(freq_lr) + freq_lr, inverse=True)

    learned_hr = np.clip(learned_hr[0].data.numpy(), 0, 1)
    bicubic_hr = bicubic_hr.data.numpy()
    hr = hr.data.numpy()

    bicubic_psnr = compare_psnr(hr, bicubic_hr)
    learned_psnr = compare_psnr(hr, learned_hr)

    avg_b_psnr += bicubic_psnr
    avg_l_psnr += learned_psnr
    max_b_psnr = max(max_b_psnr, bicubic_psnr)
    max_l_psnr = max(max_l_psnr, learned_psnr)

    print(f"{path.split('_SRF')[0]} Bicubic PSNR:", bicubic_psnr)
    print(f"{path.split('_SRF')[0]} Learned PSNR:", learned_psnr)

print('\n')
print("Average Bicubic PSNR:", avg_b_psnr / len(image_paths))
print("Average Learned PSNR:", avg_l_psnr / len(image_paths))
print("Max Bicubic PNSR:    ", max_b_psnr)
print("Max Learned PSNR:    ", max_l_psnr)