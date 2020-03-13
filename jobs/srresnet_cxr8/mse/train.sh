#!/bin/bash

# Not sure what this does?
# nvidia-smi

# Activate desired environment
# Need to source bashrc for some reason
source ~/.bashrc
conda activate pytorch-env

# Check to see if data is in the right spot
if [[ ! -d "/data/pacole2" ]]
then
    mkdir /data/pacole2
fi

if [[ ! -d "/data/pacole2/miniCXR8/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    cp /shared/rsaas/pacole2/miniCXR8.tar.gz /data/pacole2/

    cd /data/pacole2/
    tar -xzf miniCXR8.tar.gz
    rm miniCXR8.tar.gz
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/freq-sr/
echo "Running python script now"
python train_srresnet.py --data /data/pacole2/miniCXR8/images --dataset CXR8 --upscale 1 --content_loss mse --checkpointdir checkpoints/srresnet_cxr8/mse/ --checksample --wlmbda 1e-3
