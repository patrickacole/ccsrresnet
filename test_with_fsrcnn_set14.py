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
from utils.hartleytransform import *
from model.FSRCNN import *
from model.FreqSR import *

# global variables
M, N = (256, 256)
device = None

# load in fsrcnn as upsample
upsample = FSRCNN(scale_factor=2)
device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
checkpoint = torch.load(f'checkpoints/fsrcnn/fsrcnn_x2.pth', map_location='cpu').items()
# Model was trained with nn.DataParallel so need to get rid of `module.` in every key
state_dict = {}
for key, val in checkpoint:
    state_dict[key] = val
upsample.load_state_dict(state_dict)
upsample.eval()

# load in frequency model
model = FreqSR(shape=(1, M, N))
checkpoint = torch.load(f'checkpoints/fsrcnn_backend/best.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

data_path = "dataset/Set14/image_SRF_2/"
image_paths = os.listdir(data_path)
image_paths = [path for path in image_paths if '_HR' in path]
image_paths = sorted(image_paths)

avg_l_psnr = 0.0
avg_f_psnr = 0.0
avg_b_psnr = 0.0
max_l_psnr = 0.0
max_f_psnr = 0.0
max_b_psnr = 0.0
for path in image_paths:
    hr = Image.open(os.path.join(data_path, path)).convert('RGB')
    if (hr.size != (N, M)):
        hr = hr.resize((N, M), resample=Image.LANCZOS)
    lr = hr.resize((N // 2, M // 2), resample=Image.LANCZOS)
    bicubic_hr = lr.resize((N, M), resample=Image.BICUBIC)

    fht2d = FHT2D((M,N))

    lr_y, _ = preprocess(lr, device)
    bicubic_y, _ = preprocess(bicubic_hr, device)
    hr_y, _ = preprocess(hr, device)

    fsrcnn_y = upsample(lr_y.unsqueeze(0))
    learned_y = fht2d(model(fht2d(fsrcnn_y)) + fht2d(fsrcnn_y), inverse=True)

    # convert to numpy
    bicubic_y = bicubic_y.cpu().data.numpy()
    fsrcnn_y = fsrcnn_y[0].cpu().data.numpy()
    learned_y = learned_y[0].cpu().data.numpy()
    hr_y = hr_y.cpu().data.numpy()

    learned_y = np.clip(learned_y, 0, 1)

    bicubic_psnr = compare_psnr(hr_y, bicubic_y)
    fsrcnn_psnr = compare_psnr(hr_y, fsrcnn_y)
    learned_psnr = compare_psnr(hr_y, learned_y)

    avg_b_psnr += bicubic_psnr
    avg_f_psnr += fsrcnn_psnr
    avg_l_psnr += learned_psnr
    max_b_psnr = max(max_b_psnr, bicubic_psnr)
    max_f_psnr = max(max_f_psnr, fsrcnn_psnr)
    max_l_psnr = max(max_l_psnr, learned_psnr)

    print(path)
    print(f"{path.split('_SRF')[0]} Bicubic PSNR:", bicubic_psnr)
    print(f"{path.split('_SRF')[0]} FSRCNN PSNR :", fsrcnn_psnr)
    print(f"{path.split('_SRF')[0]} Learned PSNR:", learned_psnr)

print('\n')
print("Average Bicubic PSNR:", avg_b_psnr / len(image_paths))
print("Average FSRCNN PSNR :", avg_f_psnr / len(image_paths))
print("Average Learned PSNR:", avg_l_psnr / len(image_paths))
print("Max Bicubic PNSR:    ", max_b_psnr)
print("Max FSRCNN PSNR :    ", max_f_psnr)
print("Max Learned PSNR:    ", max_l_psnr)