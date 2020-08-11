import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from argparse import ArgumentParser

# custom imports
from utils.checkpoints import *
from model.CCSRResNet import *
from model.SRResNet import *


# global variables
device = None
args = None
dataset = None

def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('sample', help="Path to sample to test")
    parser.add_argument('--checkpointdir', default="checkpoints/ccsrresnet_dl/mse/", help="Path to checkpoint directory")
    return parser.parse_args()


if __name__=="__main__":
    args = args_parse()

    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
    modelSR = CCSRResNet(nc=1, upscale=1)

    # load from checkpoint files
    load_checkpoint(os.path.join(args.checkpointdir, 'super_resolution'), 'last', modelSR)
    modelSR.to(device)

    ctslice = np.asarray(Image.open(args.sample).convert('L')) / 255.0
    ctslice_t = torch.from_numpy(ctslice).unsqueeze(0).unsqueeze(0).float()
    reconst_t = modelSR(ctslice_t)[0,0]
    reconst = reconst_t.data.numpy()

    reconst = (reconst - reconst.min()) / (reconst.max() - reconst.min())
    img = Image.fromarray((255 * reconst).astype(np.uint8))
    img.save(os.path.join(os.path.dirname(args.sample),'reconst.png'))

    plt.style.use('dark_background')
    figs, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].imshow(ctslice, cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(reconst, cmap='gray')
    axs[1].axis('off')
    plt.show()
