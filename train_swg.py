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
from model.SRResNet import SRResNet
from model.SWGDiscriminator import SWGDiscriminator


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
    parser.add_argument('--nprojections', default=10000, type=int, help="Number of projections for swg loss")
    parser.add_argument('--content_loss', default="mse", help="Content loss can currently be mse, abs, or None")
    parser.add_argument('--clmbda', default=1.0, type=float, help="Weight of content loss")
    parser.add_argument('--wlmbda', default=1e-3, type=float, help="Weight of sliced wasserstein loss")
    parser.add_argument('--checksample', default=False, action='store_true', help="Whether to save an image intermediately throughout training")
    parser.add_argument('--checkpointdir', default="checkpoints/swg/srresnet/", help="Path to checkpoint directory")
    return parser.parse_args()

def psnr(learned, real):
    learned = torch.clamp(learned, min=0, max=1)
    mse = ((learned - real) ** 2).view(real.size(0), -1).mean(dim=-1)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr

def wasserstein2d(fake_feats, real_feats, fdim):
    # Wasserstein-2 distance
    # fdim - feature dimensions in each sample
    # calculate theta for random projections
    theta = torch.randn((fdim, args.nprojections),
                        requires_grad=False,
                        device=device)
    # normalize theta with l2 norm
    theta = F.normalize(theta, dim=0, p=2)

    # calculate fake and real projections
    fake_projs = fake_feats @ theta
    real_projs = real_feats.detach() @ theta

    return torch.mean((fake_projs - real_projs) ** 2) # don't believe sorting will work because will match incorrect pairs

    '''
    real_sorted, real_indices = torch.topk(real_projs, args.batch, dim=0)
    fake_sorted, fake_indices = torch.topk(fake_projs, args.batch, dim=0)

    # For faster gradient computation, we do not use sorted_fake to compute
    # loss. Instead we re-order the sorted_true so that the samples from the
    # true distribution go to the correct sample from the fake distribution.
    flat_real = real_sorted.view(-1)

    # Modify the indices to reflect this transition to an array.
    # new index = row + index
    rows = torch.tensor([args.batch * np.floor(i * 1.0 / args.batch) for i in range(args.nprojections * args.batch)])
    rows = rows.type(torch.int32)
    rows = rows.to(device)

    flat_idx = fake_indices.view(-1) + rows.view(-1)
    rearranged_real = torch.zeros_like(flat_real).scatter_(0, flat_idx, flat_real)
    rearranged_real = rearranged_real.view((args.batch, args.nprojections))
    rearranged_real = rearranged_real.to(device)

    return torch.mean((fake_projs - rearranged_real) ** 2)
    '''

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
    c_criterion = None
    if args.content_loss == 'mse':
        c_criterion = nn.MSELoss()
    elif args.content_loss == 'abs':
        c_criterion = nn.L1Loss()
    elif args.content_loss != 'None':
        raise NotImplementedError('The loss provided has not been implemented yet')

    # loss function for sliced wasserstein
    swg_criterion = nn.BCEWithLogitsLoss()

    # get dimension of features from discriminator
    ## try-except used to account for dataparallel wrapper
    try:
        fdim = modelD.fdim
    except:
        fdim = modelD.module.fdim

    # calculate when to print each epoch
    print_idx = len(dataloader) // args.num_epoch_prints

    for e in range(args.epochs):
        print("Epoch [{} / {}]".format(e + 1, args.epochs))

        # keep track of the avg loss values
        avg_content_loss = 0.0
        avg_swg_loss = 0.0
        avg_fake_d_loss = 0.0
        avg_real_d_loss = 0.0

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
            real_d_loss = swg_criterion(real_label, torch.ones_like(real_label))
            avg_real_d_loss += real_d_loss.item()

            ## fake data
            with torch.no_grad():
                fake_data = modelSR(imageLR)
            fake_label = modelD(fake_data.detach())
            fake_d_loss = swg_criterion(fake_label, torch.zeros_like(fake_label))
            avg_fake_d_loss += fake_d_loss.item()

            ## sliced wasserstein gan loss
            swg_loss = fake_d_loss + real_d_loss
            avg_swg_loss += swg_loss.item()

            ## discriminator loss
            d_loss = swg_loss

            ## calculate gradients
            d_loss.backward()

            ## update parameters
            optimizerD.step()

            ## remove variables to release some memory on gpu
            del d_loss, swg_loss, fake_d_loss, real_d_loss
            del fake_label, fake_data, real_label, real_data

            ##################################
            # train super resolution network
            ##################################
            for param in modelD.parameters():
                param.requires_grad_(False)

            optimizerSR.zero_grad()

            ## generate fake data
            fake_data = modelSR(imageLR)

            ## get fake features
            _, fake_feats = modelD(fake_data, features=True)

            ## get real features
            _, real_feats = modelD(imageHR, features=True)

            ## calculate wasserstein loss
            swg_loss = wasserstein2d(fake_feats, real_feats, fdim)
            avg_swg_loss += swg_loss.item()

            ## calculate content loss
            content_loss = None
            if args.content_loss == 'mse' or args.content_loss == 'abs':
                content_loss = c_criterion(fake_data, imageHR)
            elif args.content_loss == 'None':
                content_loss = torch.tensor(0.0).to(device)
            avg_content_loss += content_loss.item()

            ## calculate gradient
            sr_loss = args.clmbda * content_loss + args.wlmbda * swg_loss
            sr_loss.backward()

            ## update parameters
            optimizerSR.step()

            ## calculate psnr
            learned_psnr = psnr(fake_data, imageHR)
            avg_psnr += learned_psnr.mean().item()

            ## remove variables to release some memory on gpu
            del sr_loss, content_loss, swg_loss, fake_data, fake_feats, real_feats
            del imageLR, imageHR

            ## print status of training
            if (i + 1) % print_idx == 0:
                print("===> Epoch [{} / {}]: Batch [{} / {}]:".format(e + 1, args.epochs, i + 1, len(dataloader)),
                      "D Fake Loss: {:.4f},".format(avg_fake_d_loss / (i + 1)),
                      "D Real Loss: {:.4f},".format(avg_real_d_loss / (i + 1)),
                      "SWG Loss: {:.4f},".format(avg_swg_loss / (i + 1)),
                      "Content Loss: {:.4f},".format(avg_content_loss / (i + 1)),
                      "PSNR: {:.2f}".format(avg_psnr / (i + 1)))

        # take a step with the lr schedulers
        lrschedulerSR.step()
        lrschedulerD.step()

        # print final results of the training run
        print("Epoch [{} / {}]:".format(e + 1, args.epochs),
              "D Fake Loss: {:.4f},".format(avg_fake_d_loss / len(dataloader)),
              "D Real Loss: {:.4f},".format(avg_real_d_loss / len(dataloader)),
              "SWG Loss: {:.4f},".format(avg_swg_loss / len(dataloader)),
              "Content Loss: {:.4f},".format(avg_content_loss / len(dataloader)),
              "PSNR: {:.2f}".format(avg_psnr / len(dataloader)))

        # check to save sample, only do every 50 epochs
        # TODO: remove the e == 0 condition
        if args.checksample and ((e + 1) % 50 == 0 or e == 0):
            with torch.no_grad():
                learned = modelSR(test_images.to(device))

            # convert to numpy array
            learned = learned.cpu().data.numpy()

            # if output directory is not made create one
            savedir = 'swg'
            if args.content_loss != 'None':
                savedir = args.content_loss
            savedir = os.path.join('output_swg', savedir)
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
    print("Number of projections:" + str(args.nprojections))
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
    modelD  = SWGDiscriminator((1,128,128), nlayers=5)
    if (torch.cuda.device_count() > 1):
        device_ids = list(range(torch.cuda.device_count()))
        print("GPU devices being used: ", device_ids)
        modelSR = nn.DataParallel(modelSR, device_ids=device_ids)
        modelD  = nn.DataParallel(modelD, device_ids=device_ids)
    modelSR.to(device)
    modelD.to(device)

    train(modelSR, modelD, dataloader)
