import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom imports
from utils.dataset import *
from utils.checkpoints import *
from utils.hartleytransform import *
from model.SRResNet import *


# global variables
M, N = (128, 128)
device = None
args = None
test_images = None
test_names = None

# learning rate decay class
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), \
                 "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class MixedLoss():
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.mse = nn.MSELoss()
        self.fht2d = FHT2D((M,N))
        self.wl2 = weightedEuclideanLoss

    def __call__(self, learned, real, epoch):
        alpha = 1.0 - epoch / self.n_epochs
        beta = 1.0 - alpha

        mse_loss = self.mse(learned, real)
        wl2_loss = self.wl2(self.fht2d(learned), self.fht2d(real))
        return alpha * mse_loss + beta * wl2_loss

def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('--data', default="../datasets/xray_images/", help="Path to where data is stored")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=500, type=int, help="Number of epochs to train")
    parser.add_argument('--num_epoch_prints', default=10, type=int, help="Number of times to print each epoch")
    parser.add_argument('--start_decay', default=400, type=int, help="Epoch to start decaying the learning rate")
    parser.add_argument('--batch', default=64, type=int, help="Batch size to use while training")
    parser.add_argument('--content_loss', default="mse", help="Content loss can currently be wl2, mse, mix, abs, or None")
    parser.add_argument('--clmbda', default=1.0, type=float, help="Weight of content loss")
    parser.add_argument('--wlmbda', default=1e-3, type=float, help="Weight of wasserstein loss")
    parser.add_argument('--checksample', default=False, action='store_true', help="Whether to save an image intermediately throughout training")
    parser.add_argument('--checkpointdir', default="checkpoints/srresnet/", help="Path to checkpoint directory")
    return parser.parse_args()

def calc_gradient_penalty(modelD, real_data, fake_data, lmbda=10):
    nc = real_data.shape[1]
    dim = real_data.shape[-1]
    alpha = torch.rand(args.batch, 1)
    alpha = alpha.expand(args.batch, int(real_data.nelement() / args.batch)).contiguous()
    alpha = alpha.view(args.batch, nc, dim, dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(args.batch, nc, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = modelD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    return gradient_penalty

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
    m = (torch.abs(M / 2 - torch.arange(M)[:,None]) / (M / 2)) ** 2
    n = (torch.abs(N / 2 - torch.arange(N)[None,:]) / (N / 2)) ** 2
    w = torch.exp(a * m + b * n).to(device)

    # take the 2 norm of the flattened image
    # this is equivalent to taking the frobenius norm of the image matrix
    d = w[None,None,:] * (learned - real)
    d = d.view(-1, c * M * N)
    l = 0.5 * torch.norm(d, p=2, dim=1)

    # return the mean over the given samples
    return torch.mean(l)

def train(modelSR, modelD, dataloader):
    # set optimizers
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizerSR = optim.Adam(modelSR.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerD  = optim.Adam(modelD.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # set learning rate scheduler
    # lrscheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2)
    lrschedulerSR = torch.optim.lr_scheduler.LambdaLR(optimizerSR, lr_lambda=LambdaLR(args.epochs, 0, args.start_decay).step)
    lrschedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=LambdaLR(args.epochs, 0, args.start_decay).step)

    # set loss function for content loss
    criterion = None
    fht2d = None
    if args.content_loss == 'mse':
        criterion = nn.MSELoss()
    elif args.content_loss == 'abs':
        criterion = nn.L1Loss()
    elif args.content_loss == 'wl2':
        criterion = weightedEuclideanLoss
        fht2d = FHT2D((M,N))
    elif args.content_loss == 'mix':
        criterion = MixedLoss(args.epochs)
    elif args.content_loss != 'None':
        raise NotImplementedError('The loss provided has not been implemented yet')

    # calculate when to print each epoch
    print_idx = len(dataloader) // args.num_epoch_prints

    for e in range(args.epochs):
        print("Epoch [{} / {}]".format(e + 1, args.epochs))

        # keep track of the avg loss values
        avg_content_loss = 0.0
        avg_wasserstein_loss = 0.0
        avg_fake_d_loss = 0.0
        avg_real_d_loss = 0.0
        avg_gradient_penalty = 0.0

        # keep track of the average psnr
        avg_psnr = 0.0

        # total number of images
        total_images = 0
        for i, (imageLR, imageHR) in enumerate(dataloader):
            imageLR = imageLR.to(device)
            imageHR = imageHR.to(device)

            ##################################
            # train discriminator
            ##################################
            for param in modelD.parameters():
                param.requires_grad_(True)

            optimizerD.zero_grad()

            ## real data
            real_data = imageHR
            real_label = modelD(real_data)
            real_d_loss = real_label.mean()
            avg_real_d_loss += real_d_loss.item()

            ## fake data
            with torch.no_grad():
                fake_data = modelSR(imageLR)
            fake_label = modelD(fake_data)
            fake_d_loss = fake_label.mean()
            avg_fake_d_loss += fake_d_loss.item()

            ## gradient penalty
            gradient_penalty = calc_gradient_penalty(modelD, real_data, fake_data, lmbda=10.0)
            avg_gradient_penalty += gradient_penalty.item()

            ## discriminator loss
            wassertein_loss = fake_d_loss - real_d_loss
            d_loss = wassertein_loss + gradient_penalty

            ## calculate gradients
            d_loss.backward()

            ## update parameters
            optimizerD.step()

            ## remove variables to release some memory on gpu
            del d_loss, wassertein_loss, gradient_penalty, fake_d_loss, real_d_loss
            del fake_label, fake_data, real_label, real_data

            ##################################
            # train super resolution network
            ##################################
            for param in modelD.parameters():
                param.requires_grad_(False)

            optimizerSR.zero_grad()

            ## generate fake data
            fake_data = modelSR(imageLR)

            ## calculate wasserstein loss
            fake_label = modelD(fake_data)
            wassertein_loss = -fake_label.mean()
            avg_wasserstein_loss += wassertein_loss.item()

            ## calculate content loss
            content_loss = None
            if args.content_loss == 'mse' or args.content_loss == 'abs':
                content_loss = criterion(fake_data, imageHR)
            elif args.content_loss == 'wl2':
                content_loss = criterion(fht2d(fake_data), fht2d(imageHR))
            elif args.content_loss == 'mix':
                content_loss = criterion(fake_data, imageHR, e)
            elif args.content_loss == 'None':
                content_loss = torch.tensor(0.0)
            avg_content_loss += content_loss.item()

            ## calculate gradient
            sr_loss = args.clmbda * content_loss + args.wlmbda * wassertein_loss
            sr_loss.backward()

            ## update parameters
            optimizerSR.step()

            ## calculate psnr
            learned_psnr = psnr(fake_data, imageHR)
            avg_psnr += learned_psnr.mean().item()

            ## remove variables to release some memory on gpu
            del sr_loss, content_loss, wassertein_loss, fake_data
            del imageLR, imageHR

            ## print status of training
            if (i + 1) % print_idx == 0:
                print("===> Epoch [{} / {}]: Batch [{} / {}]:".format(e + 1, args.epochs, i + 1, len(dataloader)),
                      "Gradient Penalty: {:.4f},".format(avg_gradient_penalty / (i + 1)),
                      "D Fake Loss: {:.4f},".format(avg_fake_d_loss / (i + 1)),
                      "D Real Loss: {:.4f},".format(avg_real_d_loss / (i + 1)),
                      "Wasserstein Loss: {:.4f},".format(avg_wasserstein_loss / (i + 1)),
                      "Content Loss: {:.4f},".format(avg_content_loss / (i + 1)),
                      "PSNR: {:.2f}".format(avg_psnr / (i + 1)))

        # take a step with the lr schedulers
        lrschedulerSR.step()
        lrschedulerD.step()

        # print final results of the training run
        print("Epoch [{} / {}]:".format(e + 1, args.epochs),
              "Gradient Penalty: {:.4f},".format(avg_gradient_penalty / len(dataloader)),
              "D Fake Loss: {:.4f},".format(avg_fake_d_loss / len(dataloader)),
              "D Real Loss: {:.4f},".format(avg_real_d_loss / len(dataloader)),
              "Wasserstein Loss: {:.4f},".format(avg_wasserstein_loss / len(dataloader)),
              "Content Loss: {:.4f},".format(avg_content_loss / len(dataloader)),
              "PSNR: {:.2f}".format(avg_psnr / len(dataloader)))

        # check to save sample, only do every 50 epochs
        if args.checksample and (e + 1) % 50 == 0:
            with torch.no_grad():
                learned = modelSR(test_images.to(device))

            # convert to numpy array
            learned = learned.cpu().data.numpy()

            # if output directory is not made create one
            savedir = 'wgan'
            if args.content_loss != 'None':
                savedir = args.content_loss
            savedir = os.path.join('output', savedir)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            for i in range(learned.shape[0]):
                # get test image, 0 index is because grayscale
                image = learned[i][0]
                image = Image.fromarray((255 * image).astype(np.uint8))
                image.save(os.path.join(savedir, test_names[i]))

        # save models
        sr_state = {'epoch' : e + 1,
                    'state_dict' : modelSR.state_dict(),
                    'optim_dict' : optimizerSR.state_dict()}
        save_checkpoint(sr_state, isbest=False,
                        checkpoint=os.path.join(args.checkpointdir, "super_resolution"))
        d_state = {'epoch' : e + 1,
                   'state_dict' : modelD.state_dict(),
                   'optim_dict' : optimizerD.state_dict()}
        save_checkpoint(d_state, isbest=False,
                        checkpoint=os.path.join(args.checkpointdir, "discriminator"))


if __name__=="__main__":
    print("Beginning training for SRResNet model...")
    args = args_parse()

    print("Using the following hyperparemters:")
    print("Data:                 " + args.data)
    print("Learning rate:        " + str(args.lr))
    print("Number of epochs:     " + str(args.epochs))
    print("Prints per epoch:     " + str(args.num_epoch_prints))
    print("Start decay:          " + str(args.start_decay))
    print("Batch size:           " + str(args.batch))
    print("Content loss:         " + args.content_loss)
    print("CLambda:              " + str(args.clmbda))
    print("WLambda:              " + str(args.wlmbda))
    print("Check Sample:         " + str(args.checksample))
    print("Checkpoint directory: " + args.checkpointdir)
    print("Cuda:                 " + str(torch.cuda.device_count()))
    print("")

    dataset = NoisyXrayDataset(args.data)
    if args.checksample:
        test_images = dataset[0][0].unsqueeze(0)
        test_names = [dataset.at(0).lstrip(os.path.join(args.data, 'train_images_64x64'))]

    dataloader = DataLoader(dataset, batch_size=args.batch,
                            shuffle=True, num_workers=8)

    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])

    modelSR = SRResNet(nc=1, upscale=2)
    modelD  = Discriminator(nc=1)
    if (torch.cuda.device_count() > 1):
        device_ids = list(range(torch.cuda.device_count()))
        print("GPU devices being used: ", device_ids)
        modelSR = nn.DataParallel(modelSR, device_ids=device_ids)
        modelD  = nn.DataParallel(modelD, device_ids=device_ids)
    modelSR.to(device)
    modelD.to(device)

    train(modelSR, modelD, dataloader)
