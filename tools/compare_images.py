import os
import numpy as np

from PIL import Image
from argparse import ArgumentParser
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# import for testing purposes
import matplotlib.pyplot as plt

# get arguments
parser = ArgumentParser(description="Arguments for comparing results of different methods")
parser.add_argument('--data', default="/Users/Patrick/Downloads/DeepLesionTestPreprocessed/miniStudies/", help="Path to true images")
parser.add_argument('--ccsrresnet', default="/Users/Patrick/Downloads/ccsrresnet-results/", help="Path to ccsrresnet results")
parser.add_argument('--wgan_vgg', default="/Users/Patrick/Downloads/wgan_vgg-results/", help="Path to wgan_vgg results")
parser.add_argument('--n3net', default="/Users/Patrick/Downloads/n3net-results/", help="Path to n3net results")
parser.add_argument('--dncnn', default="/Users/Patrick/Downloads/dncnn-results/", help="Path to dncnn results")
parser.add_argument('--bm3d', default="/Users/Patrick/Downloads/bm3d-results/sigma-20/", help="Path to bm3d results")
parser.add_argument('--random_crop', default=0, type=int, help="Size to randomly crop images")
args = parser.parse_args()

# get all samples
samples = {'GT' : [], 'CCSRResNet' : [], 'WGAN-VGG' : [], 'N3Net' : [], 'DnCNN' : [], 'BM3D' : []}

studies = os.listdir(args.data)
for study in studies:
    if os.path.isdir(study):
        continue
    for sample in os.listdir(os.path.join(args.data, study)):
        if '.png' not in sample:
            continue

        samples['GT'].append(os.path.join(args.data, study, sample))
        samples['CCSRResNet'].append(os.path.join(args.ccsrresnet, study, sample))
        samples['WGAN-VGG'].append(os.path.join(args.wgan_vgg, study, sample))
        samples['N3Net'].append(os.path.join(args.n3net, study, sample))
        samples['DnCNN'].append(os.path.join(args.dncnn, study, sample))
        samples['BM3D'].append(os.path.join(args.bm3d, study, sample))

num_samples = len(samples['GT'])

# go through all images
## create a function to load image with the cropped border
def load_image(filename, xs=0, ys=0, size=None):
    img = np.asarray(Image.open(filename).convert('L'))
    if size:
        img = img[xs:xs+size,ys:ys+size]
    return img

# get random index
idx = np.random.randint(num_samples)
print('Sample used: {}\n'.format(samples['GT'][idx].split(args.data)[1]))

# if random crop
size = args.random_crop if args.random_crop != 0 else None
xs = np.random.randint(0, 256 - args.random_crop) if size != None else 0
ys = np.random.randint(0, 256 - args.random_crop) if size != None else 0

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.canvas.set_window_title('Compare results')

for i, key in enumerate(samples.keys()):
    x = i // 3
    y = i % 3
    img = load_image(samples[key][idx], xs=xs, ys=ys, size=size)
    if i == 0:
        gt = img.copy()
        text = 'PSNR: inf SSIM: 1.0000'
    else:
        psnr = compare_psnr(gt, img, data_range=255)
        ssim = compare_ssim(gt, img, data_range=255)
        text = 'PSNR: {0:0.2f} SSIM: {1:0.4f}'.format(psnr, ssim)

    axes[x,y].imshow(img, cmap='gray')
    axes[x,y].set_title(key)
    axes[x,y].set_xticks([])
    axes[x,y].set_yticks([])
    axes[x,y].text(0.5,-0.05, text, size=8, ha="center",
                   transform=axes[x,y].transAxes)

plt.show()
