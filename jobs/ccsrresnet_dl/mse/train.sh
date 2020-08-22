#!/bin/bash

# Not sure what this does?
nvidia-smi

# Activate desired environment
# Need to source bashrc for some reason
source ~/.bashrc
conda activate pytorch-env

# Check to see if data is in the right spot
if [[ ! -d "/data/pacole2" ]]
then
    mkdir /data/pacole2
fi

if [[ ! -d "/data/pacole2/DeepLesionPreprocessed/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    cp /shared/rsaas/pacole2/DeepLesionPreprocessed.zip /data/pacole2/

    cd /data/pacole2/
    unzip -qq DeepLesionPreprocessed.zip
    rm DeepLesionPreprocessed.zip
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/freq-sr/
echo "Running python script now"
# python train_ccsrresnet.py --data /data/pacole2/DeepLesionPreprocessed/miniStudies/ --dataset DeepLesion --upscale 1 --content_loss mse --checkpointdir checkpoints/ccsrresnet_dl/mse/ --checksample --wlmbda 1e-3 --batch 32 --load
## retrain for slightly different input data
python train_ccsrresnet.py --data /data/pacole2/DeepLesionPreprocessed/miniStudies/ --dataset DeepLesion --upscale 1 --content_loss mse --checkpointdir checkpoints/ccsrresnet_dl/mse/ --checksample --wlmbda 1e-3 --batch 32 --epochs 50 --start_decay 35 --load --retrain
