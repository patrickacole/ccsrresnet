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
# from model.srresnet import _NetG as SRResNet
from model.FreqSR import *

# suppress warnings from model
import warnings

# global variables
M, N = (256, 256)
SCALE = 4


class PIL_to_tensor():
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, image):
        return self.totensor(image)


def recursion_change_bn(module):
    if isinstance(module, torch.nn.InstanceNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

# load in fsrcnn as upsample
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    upsample = torch.load(f'checkpoints/srresnet/model_srresnet.pth', map_location='cpu')["model"]
for i, (name, module) in enumerate(upsample._modules.items()):
    module = recursion_change_bn(module)
device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
upsample.eval()

# model = FreqSR(shape=(3, M, N))
# checkpoint = torch.load(f'checkpoints/{3}x{M}x{N}/best.pth', map_location='cpu')
# # Model was trained with nn.DataParallel so need to get rid of `module.` in every key
# state_dict = {}
# for key in checkpoint['state_dict'].keys():
#     state_dict[key.split('module.')[1]] = checkpoint['state_dict'][key]
# model.load_state_dict(state_dict)

data_path = "dataset/Set5/image_SRF_4/"
image_paths = os.listdir(data_path)
image_paths = [path for path in image_paths if '_HR' in path]
image_paths = sorted(image_paths)

avg_l_psnr = 0.0
avg_b_psnr = 0.0
max_l_psnr = 0.0
max_b_psnr = 0.0
for path in image_paths:
    hr = Image.open(os.path.join(data_path, path)).convert('RGB')
    if (hr.size != (N, M)):
        hr = hr.resize((N, M), resample=Image.LANCZOS)
    lr = hr.resize((N // SCALE, M // SCALE), resample=Image.LANCZOS)
    bicubic_hr = lr.resize((N, M), resample=Image.BICUBIC)

    convert = PIL_to_tensor()
    fht2d = FHT2D((M,N))
    hr = convert(hr)
    lr = convert(lr)
    bicubic_hr = convert(bicubic_hr)

    learned_hr = upsample(lr.unsqueeze(0))

    learned_hr = np.clip(learned_hr[0].data.numpy(), 0, 1)
    bicubic_hr = bicubic_hr.data.numpy()
    hr = hr.data.numpy()

    bicubic_y = convert_rgb_to_y(255 * bicubic_hr) / 255.0
    learned_y = convert_rgb_to_y(255 * learned_hr) / 255.0
    hr_y = convert_rgb_to_y(255 * hr) / 255.0

    bicubic_psnr = compare_psnr(hr_y, bicubic_y)
    learned_psnr = compare_psnr(hr_y, learned_y)

    avg_b_psnr += bicubic_psnr
    avg_l_psnr += learned_psnr
    max_b_psnr = max(max_b_psnr, bicubic_psnr)
    max_l_psnr = max(max_l_psnr, learned_psnr)

    print(path)
    print(f"{path.split('_SRF')[0]} Bicubic PSNR:", bicubic_psnr)
    print(f"{path.split('_SRF')[0]} SRResNet PSNR:", learned_psnr)

print('\n')
print("Average Bicubic PSNR:", avg_b_psnr / len(image_paths))
print("Average SRResNet PSNR:", avg_l_psnr / len(image_paths))
print("Max Bicubic PNSR:    ", max_b_psnr)
print("Max SRResNet PSNR:    ", max_l_psnr)