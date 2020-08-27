import os
import numpy as np

from PIL import Image
from tqdm import tqdm
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
parser.add_argument('--crop_border', default=0, type=int, help="Number of pixels on the border to remove")
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
def load_image(filename, cb=0):
    img = np.asarray(Image.open(filename).convert('L'))
    if cb > 0:
        img = img[cb:-cb, cb:-cb]
    return img

# img = load_image(samples['GT'][0], cb=args.crop_border)
# print(img.shape)
# plt.imshow(img, cmap='gray')
# plt.show()

psnrs = {'CCSRResNet' : [], 'WGAN-VGG' : [], 'N3Net' : [], 'DnCNN' : [], 'BM3D' : []}
ssims = {'CCSRResNet' : [], 'WGAN-VGG' : [], 'N3Net' : [], 'DnCNN' : [], 'BM3D' : []}

with tqdm(total=num_samples, ncols=175,
    bar_format="{n_fmt} / {total_fmt} [{bar}]" + \
    " - {postfix[0]}: {postfix[CCSRResNet]}" + \
    " - {postfix[1]}: {postfix[WGAN-VGG]}" + \
    " - {postfix[2]}: {postfix[N3Net]}" + \
    " - {postfix[3]}: {postfix[DnCNN]}" + \
    " - {postfix[4]}: {postfix[BM3D]}",
    postfix={0:'CCSRResNet', 1:'WGAN-VGG', 2:'N3Net', 3:'DnCNN', 4:'BM3D', \
            'CCSRResNet':'...', 'WGAN-VGG':'...', 'N3Net':'...', 'DnCNN':'...', 'BM3D':'...'}) as t:

    for i in range(num_samples):
        gt = load_image(samples['GT'][i], cb=args.crop_border)
        for key in samples.keys():
            if key == 'GT':
                continue
            sample = load_image(samples[key][i], cb=args.crop_border)

            psnrs[key].append(compare_psnr(gt, sample, data_range=255))
            ssims[key].append(compare_ssim(gt, sample, data_range=255))

            t.postfix[key] = '[{0:0.2f}, {1:0.4f}]'.format(np.mean(psnrs[key]), np.mean(ssims[key]))

        t.update(1)

