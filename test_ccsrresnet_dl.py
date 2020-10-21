import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# custom imports
from utils.dataset import *
from utils.checkpoints import *
from model.CCSRResNet import *


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
    parser.add_argument('--data', default="../../Downloads/DeepLesion2/miniStudies/", help="Path to where data is stored")
    parser.add_argument('--batch', default=32, type=int, help="Batch size to use while training")
    parser.add_argument('--batch_break', default=-1, type=int, help="How many batches to go through, default is all")
    parser.add_argument('--upscale', default=2, type=int, help="Amount to upscale by")
    parser.add_argument('--num_prints', default=8, type=int, help="Number of times in the testing loop")
    parser.add_argument('--metric', default="all", help="Metric to evaluate model on test data. psnr, ssim, rmse, or all")
    parser.add_argument('--savedir', default="output/ccsrresnet-results/", help="Path to store generated super resolution files")
    parser.add_argument('--checkpointdir', default="checkpoints/ccsrresnet/", help="Path to checkpoint directory")
    return parser.parse_args()

# def calc_psnr(learned, real):
#     learned = torch.clamp(learned, min=0, max=1)
#     mse = ((learned - real) ** 2).view(real.size(0), -1).mean(dim=-1)
#     psnr = 10.0 * torch.log10(1.0 / mse)
#     return psnr

def calc_psnr(learned, real, data_range=1.0):
    learned = learned.data.cpu().numpy().astype(np.float32)
    real = real.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(learned.shape[0]):
        psnr += compare_psnr(real[i,:,:,:], learned[i,:,:,:], data_range=data_range)
    return (psnr / learned.shape[0])

def calc_ssim(learned, real, data_range=1.0):
    learned = learned.data.cpu().numpy().astype(np.float32)
    real = real.data.cpu().numpy().astype(np.float32)
    ssim = 0
    for i in range(learned.shape[0]):
        ssim += compare_ssim(real[i,0,:,:], learned[i,0,:,:], data_range=data_range)
    return (ssim / learned.shape[0])

def calc_rsme(learned, real):
    learned = torch.clamp(learned, min=0, max=1)
    mse = ((255.0 * (learned - real)) ** 2).view(real.size(0), -1).mean(dim=-1)
    rmse = torch.sqrt(mse)
    return rmse

def test(modelSR, dataloader):
    criterion = None
    if args.metric == 'psnr':
        criterion = calc_psnr
    elif args.metric == 'ssim':
        criterion = calc_ssim
    elif args.metric == 'rmse':
        criterion = calc_rsme
    elif args.metric == 'all':
        criterion = [calc_psnr, calc_ssim, calc_rsme]
        criterion_names = ['psnr', 'ssim', 'rmse']
    else:
        raise NotImplementedError('The metric provided has not been implemented yet')

    # calculate when to print each epoch
    print_idx = len(dataloader) // args.num_prints

    if args.metric == 'all':
        avg_score = [0.0] * len(criterion)
    else:
        avg_score = 0.0
    total_images = 0
    print("Beginning testing loop...")
    for i, (imageLR, imageHR) in enumerate(dataloader):
        imageLR = imageLR.to(device)
        imageHR = imageHR.to(device)

        with torch.no_grad():
            learnedHR = modelSR(imageLR)

        # save images
        for j in range(imageLR.size(0)):
            # save generated
            image = learnedHR[j][0]
            image = image.cpu().data.numpy()
            filepath = dataset.at(total_images + j).lstrip(args.data)
            if not os.path.exists(os.path.dirname(os.path.join(args.savedir, filepath))):
                os.makedirs(os.path.dirname(os.path.join(args.savedir, filepath)))
            image = Image.fromarray((255 * image).astype(np.uint8))
            image.save(os.path.join(args.savedir, filepath))

            # save low res
            ## don't need to save low res
            # image = imageLR[j][0]
            # image = image.cpu().data.numpy()
            # filepath = 'noise_' + dataset.at(total_images + j).lstrip(args.data)
            # if not os.path.exists(os.path.dirname(os.path.join(args.savedir, filepath))):
            #     os.makedirs(os.path.dirname(os.path.join(args.savedir, filepath)))
            # image = Image.fromarray((255 * image).astype(np.uint8))
            # image.save(os.path.join(args.savedir, filepath))

        if args.metric == 'all':
            score = [c(learnedHR, imageHR) for c in criterion]
            for j in range(len(score)):
                avg_score[j] += score[j].sum().item()
        else:
            score = criterion(learnedHR, imageHR)
            avg_score += score.sum().item()
        total_images += imageLR.size(0)

        if (i + 1) % print_idx == 0:
            if args.metric == 'psnr':
                print('===> Batch: [{} / {}] Average psnr: {:.2f}'.format(i + 1, len(dataloader), avg_score / total_images))
            elif args.metric == 'ssim':
                print('===> Batch: [{} / {}] Average ssim: {:.2f}'.format(i + 1, len(dataloader), avg_score / total_images))
            elif args.metric == 'rmse':
                print('===> Batch: [{} / {}] Sum rmse: {:.2f}'.format(i + 1, len(dataloader), avg_score))
            elif args.metric == 'all':
                print('===> Batch: [{} / {}] Average psnr: {:.2f} Average ssim: {:.2f} Average rmse: {:.2f}'.format(i + 1, len(dataloader), avg_score[0] / total_images, avg_score[1] / total_images, avg_score[2]))

        args.batch_break -= 1
        if args.batch_break == 0:
            break

    if args.metric == 'all':
        avg_score = [score / total_images for score in avg_score]
    else:
        avg_score /= total_images

    if args.metric == 'psnr':
        print('Model had an average psnr of {:.2f} on the test dataset'.format(avg_score))
    elif args.metric == 'ssim':
        print('Model had an average ssim of {:.2f} on the test dataset'.format(avg_score))
    elif args.metric == 'rmse':
        print('Model had a sum rmse of {:.2f} on the test dataset'.format(avg_score * total_images))
    elif args.metric == 'all':
        print('Model had an average psnr of {:.2f}, an average ssim of {:.2f}, and a sum rmse of {:.2f} on the test dataset'.format(avg_score[0], avg_score[1], avg_score[2] * total_images))
        with open(os.path.join(args.savedir, 'results.txt'), 'w') as fptr:
            fptr.write('Model had an average psnr of {:.2f}, an average ssim of {:.2f}, and a sum rmse of {:.2f} on the test dataset'.format(avg_score[0], avg_score[1], avg_score[2] * total_images))

if __name__=="__main__":
    print("Beginning testing for CCSRResNet model...")
    args = args_parse()

    print("Using the following hyperparemters:")
    print("Data:                 " + args.data)
    print("Batch size:           " + str(args.batch))
    print("Batch break:          " + str(args.batch_break))
    print("Upscale:              " + str(args.upscale))
    print("Prints:               " + str(args.num_prints))
    print("Metric:               " + args.metric)
    print("Save directory:       " + args.savedir)
    print("Checkpoint directory: " + args.checkpointdir)
    print("Cuda:                 " + str(torch.cuda.device_count()))
    print("")

    dataset = DeepLesionDataset(args.data, preprocessed=True)
    dataloader = DataLoader(dataset, batch_size=args.batch,
                            shuffle=False, num_workers=8)

    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])

    modelSR = CCSRResNet(nc=1, upscale=args.upscale)

    # load from checkpoint files
    load_checkpoint(os.path.join(args.checkpointdir, 'super_resolution'), 'last', modelSR)

    modelSR.to(device)

    # check to make sure output directory is made
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    test(modelSR, dataloader)