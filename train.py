import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom imports
from utils.dataset import *
from utils.checkpoints import *
from utils.hartleytransform import *
from model.FreqSR import *

# global variables
M, N = (256, 256)
Mx, Nx = (512, 512)
device = None
args = None

def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('--data', default="../datasets/CXR8/images/", help="Path to where data is stored")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=200, type=int, help="Number of epochs to train")
    parser.add_argument('--batch', default=32, type=int, help="Batch size to use while training")
    parser.add_argument('--checkpointdir', default="checkpoints/bicubic_backend", help="Path to checkpoint directory")
    return parser.parse_args()

def psnr(learned, real):
    learned = torch.clamp(learned, min=0, max=1)
    mse = ((learned - real) ** 2).view(real.size(0), -1).mean(dim=-1)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr

def weightedEuclideanLoss(learned, real, a=1.0, b=1.0):
    # get number of channels
    c = learned.size(1)

    # construct weighted matrix
    # this matrix puts an emphasis on the corners of the image
    m = (torch.abs(Mx / 2 - torch.arange(Mx)[:,None]) / (Mx / 2)) ** 2
    n = (torch.abs(Nx / 2 - torch.arange(Nx)[None,:]) / (Nx / 2)) ** 2
    w = torch.exp(a * m + b * n).to(device)

    # take the 2 norm of the flattened image
    # this is equivalent to taking the frobenius norm of the image matrix
    d = w[None,None,:] * (learned - real)
    d = d.view(-1, c * Mx * Nx)
    l = 0.5 * torch.norm(d, p=2, dim=1)

    # return the mean over the given samples
    return torch.mean(l)

def L2Loss(learned, real):
    c = learned.size(1)
    diff = learned - real
    diff = diff.view(-1, c * Mx * Nx)
    norm = torch.norm(diff, p=2, dim=1)
    return torch.mean(norm)

def train(model, dataloader, scale_factor=2):
    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = weightedEuclideanLoss
    lrscheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2)
    fht2d = FHT2D((Mx,Nx))
    w_pad = abs(Mx - M) // 2
    h_pad = abs(Nx - N) // 2

    best_psnr = 0
    for e in range(args.epochs):
        train_loss = 0.0
        train_psnr = 0.0
        bicubic_psnr = 0.0
        total_images = 0
        for i, (imageLR, imageHR) in enumerate(dataloader):
            imageLR = F.pad(imageLR.to(device), (w_pad, w_pad, h_pad, h_pad))
            imageHR = F.pad(imageHR.to(device), (w_pad, w_pad, h_pad, h_pad))

            # calculate residual
            imageResidual = imageHR - imageLR

            freqLR = Variable(fht2d(imageLR))
            freqResidual = Variable(fht2d(imageResidual))

            optimizer.zero_grad()
            learnedResidual = model(freqLR)
            loss = criterion(learnedResidual, freqResidual)
            loss.backward()

            # clip gradient
            lr = args.lr
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break

            clip_value = 1e4 / lr # increased from 1e3
            nn.utils.clip_grad_value_(model.parameters(), clip_value)

            optimizer.step()

            learnedHR = fht2d(learnedResidual, inverse=True) + imageLR
            learnedHR = learnedHR[:,:,w_pad:-w_pad,h_pad:-h_pad]
            imageLR = imageLR[:,:,w_pad:-w_pad,h_pad:-h_pad]
            imageHR = imageHR[:,:,w_pad:-w_pad,h_pad:-h_pad]

            bicubicHR = imageLR

            total_images += imageLR.size(0)
            train_loss += loss.detach().item()
            train_psnr += psnr(learnedHR, imageHR).sum().item()
            bicubic_psnr += psnr(bicubicHR, imageHR).sum().item()

            if (i + 1) % 100 == 0:
                print("Norm residual learned: {}".format(torch.norm(learnedResidual.view(-1, 1 * M * N), dim=1).mean()))

            if (i + 1) % 25 == 0:
                print("Epoch [{} / {}]: Batch: [{} / {}]: Avg Training Loss: {:0.4f}, Avg Training PSNR: {:0.2f}, Avg Bicubic PSNR: {:0.2f}" \
                      .format(e + 1, args.epochs, i + 1, len(dataloader), train_loss / (i + 1), train_psnr / total_images,
                              bicubic_psnr / total_images))

            del loss, imageLR, imageHR, imageResidual, freqLR, freqResidual, learnedResidual, learnedHR

        lrscheduler.step()

        print("Epoch [{} / {}]: Final Avg Training Loss: {:0.4f}, Final Avg Training PSNR: {:0.2f}, Final Avg Bicubic PSNR: {:0.2f}" \
                      .format(e + 1, args.epochs, train_loss / len(dataloader), train_psnr / total_images,
                              bicubic_psnr / total_images))

        avg_train_psnr = train_psnr / len(dataloader)
        isbest = avg_train_psnr > best_psnr
        if isbest:
            best_psnr = avg_train_psnr

        training_state = {'epoch' : e + 1,
                          'state_dict' : model.state_dict(),
                          'optim_dict' : optimizer.state_dict()}
        save_checkpoint(training_state, isbest=isbest,
                        checkpoint=args.checkpointdir)


if __name__=="__main__":
    print("Beginning training for FreqSR model...")
    args = args_parse()

    print("Using the following hyperparemters:")
    print("Data:                 " + args.data)
    print("Image size:           " + str(M) + " x " + str(N))
    print("Learning rate:        " + str(args.lr))
    print("Number of Epochs:     " + str(args.epochs))
    print("Batch size:           " + str(args.batch))
    print("Checkpoint directory: " + args.checkpointdir)
    print("Cuda:                 " + str(torch.cuda.device_count()))
    print("")

    dataset = CXR8Dataset(args.data, image_shape=(M, N), color="grayscale", upsample='bicubic')
    dataloader = DataLoader(dataset, batch_size=args.batch,
                            shuffle=True, num_workers=8)

    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])

    model = FreqSR(shape=(1, Mx, Nx))
    if (torch.cuda.device_count() > 1):
        device_ids = list(range(torch.cuda.device_count()))
        print("GPU devices being used: ", device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    train(model, dataloader)
