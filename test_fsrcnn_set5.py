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
from utils.checkpoints import *
from utils.image_utils import *
from model.FSRCNN import *

# global variables
M, N = (256, 256)
device = None


model = FSRCNN(scale_factor=2)
device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
checkpoint = torch.load(f'checkpoints/fsrcnn/fsrcnn_x2.pth', map_location='cpu').items()
# Model was trained with nn.DataParallel so need to get rid of `module.` in every key
state_dict = {}
for key, val in checkpoint:
    state_dict[key] = val
model.load_state_dict(state_dict)
model.eval()

data_path = "dataset/Set5/image_SRF_2/"
image_paths = os.listdir(data_path)
image_paths = [path for path in image_paths if '_HR' in path]
image_paths = sorted(image_paths)

avg_l_psnr = 0.0
avg_b_psnr = 0.0
max_l_psnr = 0.0
max_b_psnr = 0.0
for path in image_paths:
    hr = Image.open(os.path.join(data_path, path))
    if (hr.size != (N, M)):
        hr = hr.resize((N, M), resample=Image.LANCZOS)
    lr = hr.resize((N // 2, M // 2), resample=Image.LANCZOS)
    bicubic_hr = lr.resize((N, M), resample=Image.BICUBIC)

    lr_y, _ = preprocess(lr, device)
    bicubic_y, _ = preprocess(bicubic_hr, device)
    hr_y, _ = preprocess(hr, device)

    learned_y = model(lr_y.unsqueeze(0))


    # convert to numpy
    bicubic_y = bicubic_y.cpu().data.numpy()
    learned_y = learned_y[0].cpu().data.numpy()
    hr_y = hr_y.cpu().data.numpy()

    learned_y = np.clip(learned_y, 0, 1)

    bicubic_psnr = compare_psnr(hr_y, bicubic_y)
    learned_psnr = compare_psnr(hr_y, learned_y)

    avg_b_psnr += bicubic_psnr
    avg_l_psnr += learned_psnr
    max_b_psnr = max(max_b_psnr, bicubic_psnr)
    max_l_psnr = max(max_l_psnr, learned_psnr)

    print(path)
    print(f"{path.split('_SRF')[0]} Bicubic PSNR:", bicubic_psnr)
    print(f"{path.split('_SRF')[0]} Learned PSNR:", learned_psnr)

print('\n')
print("Average Bicubic PSNR:", avg_b_psnr / len(image_paths))
print("Average Learned PSNR:", avg_l_psnr / len(image_paths))
print("Max Bicubic PNSR:    ", max_b_psnr)
print("Max Learned PSNR:    ", max_l_psnr)