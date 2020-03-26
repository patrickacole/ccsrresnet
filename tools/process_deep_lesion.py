import os
import warnings
import numpy as np
import scipy.ndimage as ndimage

from PIL import Image
from scipy import io
from argparse import ArgumentParser
from skimage.transform import radon, iradon

# get the path for the DeepLesion dataset
parser = ArgumentParser(description="Arguments for processing DeepLesion dataset")
parser.add_argument('data', default="/data/pacole2/DeepLesion/", help="Path to where data is stored")
parser.add_argument('--image_size', default=None, nargs=2, type=int, help="Size of the image")
args = parser.parse_args()

if not os.path.exists(args.data) or not os.path.exists(os.path.join(args.data, 'Images_png')):
    raise IOError('The data path given doesn\'t exist')

# get all of the data
datapath = os.path.join(args.data, 'Images_png')
studies = [d for d in os.listdir(datapath) if '.DS' not in d and os.path.isdir(os.path.join(datapath, d))]
samples = []
for d in studies:
    samples += [os.path.join(d, f) for f in os.listdir(os.path.join(datapath, d)) if '.png' in f]

# go through all slices
for i, sample in enumerate(samples, 1):
    samplepath = os.path.join(datapath, sample)

    # load image
    image = np.asarray(Image.open(samplepath), dtype=np.int32) - 32768
    image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

    # downsample image
    if args.image_size is not None:
        image_size = np.asarray(args.image_size)
        image = Image.fromarray(image)
        image = image.resize(image_size[::-1], resample=Image.LANCZOS)
        image = np.asarray(image)

    # save output image
    savepath = samplepath.replace('Images_png', 'miniStudies').rstrip(os.path.basename(samplepath))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    image = Image.fromarray(image)
    image.save(samplepath.replace('Images_png', 'miniStudies'))

    # make noisy image
    image = np.asarray(image)
    h, w = image.shape

    ## create a sinogram
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=UserWarning)
        sino = radon(image, theta=theta, circle=True)

    ## create a mask to randomly reduce frequency of some angles
    mask = np.random.binomial(1, 0.98, sino.shape)
    nsino = np.random.poisson(sino * mask / 2)

    ## create noisy image
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=UserWarning)
        noise = iradon(nsino, theta=theta, circle=True)
    ### subtract off the offset
    noise = noise - noise[0, 0]
    ### put in range 0-255 and linear interpolate between original and radon noise one
    noise = (255 * ((noise - noise.min()) / (noise.max() - noise.min())))
    noise = 1.0 * image + 0.333 * noise
    ### put in range 0-255 and then linear interpolate between poisson of radon noise and radon noise
    noise = (255 * ((noise - noise.min()) / (noise.max() - noise.min()))).astype(np.uint8)
    noise = 0.666 * noise + 0.333 * (np.random.poisson(noise) / 255.0)

    # save input noise image
    savepath = samplepath.replace('Images_png', 'noiseStudies').rstrip(os.path.basename(samplepath))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    noise = (255 * ((noise - noise.min()) / (noise.max() - noise.min()))).astype(np.uint8)
    ## smooth
    noise = ndimage.gaussian_filter(noise, sigma=0.5)
    image = Image.fromarray(noise)
    image.save(samplepath.replace('Images_png', 'noiseStudies'))

    # check if need to print
    if i % 1000 == 0:
        print("{} / {}".format(i, len(samples)))
