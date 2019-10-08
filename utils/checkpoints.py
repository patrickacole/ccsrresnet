import os
import shutil
import torch


# Functions in this file are inspired by the following:
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

def save_checkpoint(state, isbest, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    @param state     : contains model's state_dict, may contain other keys such as epoch, optimizer state_dict (dict)
    @param isbest   : True if it is the best model seen till now (bool)
    @param checkpoint: folder where parameters are to be saved (string)
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if isbest:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpointdir, prefix, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    @param checkpointdir: directory with checkpoint files (string)
    @param prefix       : file prefix for parameters to load (string)
    @param model        : model for which the parameters are loaded (DeepConvNet)
    @param optimizer    : resume optimizer from checkpoint (optim)
    """
    checkpoint = os.path.join(checkpointdir, prefix + '.pth.tar')
    if not os.path.exists(checkpoint):
        raise IOError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
