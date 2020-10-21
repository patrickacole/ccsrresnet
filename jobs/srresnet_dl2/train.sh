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
    echo "Train data is not on gpu storage"
    echo "Copying over train data from shared storage"
    cp /shared/rsaas/pacole2/DeepLesionPreprocessed.zip /data/pacole2/

    cd /data/pacole2/
    unzip -qq DeepLesionPreprocessed.zip
    rm DeepLesionPreprocessed.zip
    cd /home/pacole2/
fi

if [[ ! -d "/data/pacole2/DeepLesionTestPreprocessed/" ]]
then
    echo "Test data is not on gpu storage"
    echo "Copying over test data from shared storage"
    cp /shared/rsaas/pacole2/DeepLesionTestPreprocessed.zip /data/pacole2/

    cd /data/pacole2/
    unzip -qq DeepLesionTestPreprocessed.zip
    rm DeepLesionTestPreprocessed.zip
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/freq-sr/
echo "Running python script now"
python -u train_srresnet2.py --data /data/pacole2/DeepLesionPreprocessed/miniStudies/ --tdata /data/pacole2/DeepLesionTestPreprocessed/miniStudies/ --dataset DeepLesion --upscale 1 --content_loss mse --checkpointdir checkpoints/srresnet_dl2/mse/ --checksample --wlmbda 1e-3 --batch 64 --epochs 300 --start_decay 250 --load