import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom imports
from utils.dataset import *
from utils.checkpoints import *
from model.FreqSR import *

# global variables
device = None
args = None

def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('--data', default="dataset/VOC2012/JPEGImages/", help="Path to where data is stored")
    parser.add_argument('--lr', default=1e-2, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train")
    parser.add_argument('--batch', default=128, type=int, help="Batch size to use while training")
    parser.add_argument('--checkpointdir', default="checkpoints/", help="Path to checkpoint directory")
    return parser.parse_args()

def psnr(learned, real):
    mse = ((learned - real) ** 2).view(real.size(0), -1).mean(dim=-1)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr

def train(model, dataloader, scale_factor=2):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.MSELoss()
    lrscheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.2)
    upscale = lambda x : F.interpolate(x, scale_factor=(scale_factor, scale_factor),
                                       mode="bilinear", align_corners=False)

    best_loss = np.inf
    for e in range(args.epochs):
        train_loss = 0.0
        train_psnr = 0.0
        bilinear_psnr = 0.0
        total_images = 0
        for i, (imageLR, imageHR) in enumerate(dataloader):
            imageLR = Variable(imageLR, requires_grad=True).to(device)
            imageHR = Variable(imageHR, requires_grad=True).to(device)

            optimizer.zero_grad()
            learnedHR = model(imageLR)
            loss = criterion(learnedHR, imageHR)
            loss.backward()
            optimizer.step()

            bilinearHR = upscale(imageLR)

            total_images += imageLR.size(0)
            train_loss += loss.item()
            train_psnr += psnr(learnedHR, imageHR).sum()
            bilinear_psnr += psnr(bilinearHR, imageHR).sum()
            del imageLR, imageHR, learnedHR

            if i % 5 == 0:
                print("Epoch [{} / {}]: Batch: [{} / {}]: Avg Training Loss: {:0.4f}, Avg Training PSNR: {:0.2f}, Avg Bilinear PSNR: {:0.2f}" \
                      .format(e + 1, args.epochs, i + 1, len(dataloader), train_loss / (i + 1), train_psnr / total_images,
                              bilinear_psnr / total_images))

        lrscheduler.step()

        print("Epoch [{} / {}]: Final Avg Training Loss: {:0.4f}, Final Avg Training PSNR: {:0.2f}, Final Avg Bilinear PSNR: {:0.2f}" \
                      .format(e + 1, args.epochs, train_loss / len(dataloader), train_psnr / total_images,
                              bilinear_psnr / total_images))

        avg_train_loss = train_loss / len(dataloader)
        isbest = avg_train_loss < best_loss
        if isbest:
            best_loss = avg_train_loss

        training_state = {'epoch' : e + 1,
                          'state_dict' : model.state_dict(),
                          'optim_dict' : optimizer.state_dict()}
        save_checkpoint(training_state, isbest=isbest,
                        checkpoint=args.checkpointdir)


if __name__=="__main__":
    print("Beginning training for FreqSR model...")

    args = args_parse()
    dataset = VOC2012(args.data, image_shape=(128, 128), grayscale=True)
    dataloader = DataLoader(dataset, batch_size=args.batch,
                            shuffle=True, num_workers=8)

    device = torch.device(("cpu","cuda")[torch.cuda.is_available()])
    model = FreqSR(shape=(1, 64, 64)).to(device)

    train(model, dataloader)
