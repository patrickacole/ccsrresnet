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

if [[ ! -d "/data/pacole2/VOCdevkit" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    cp /shared/rsaas/pacole2/VOCtrainval_11-May-2012.tar /data/pacole2/
    cd /data/pacole2/
    tar -xf VOCtrainval_11-May-2012.tar
    cd /home/pacole2/
fi

cd ~/Projects/freq-sr/dataset/ 
# Make the symbolic link to the data
if [[ -L "VOC2012" ]]
then
    unlink VOC2012
fi

ln -s /data/pacole2/VOCdevkit/VOC2012 VOC2012

# Data is ready now run python file
cd ~/Projects/freq-sr/
echo "Running python script now"
python train.py --rgb

