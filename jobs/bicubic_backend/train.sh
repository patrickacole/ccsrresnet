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

if [[ ! -d "/data/pacole2/CXR8/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    mkdir /data/pacole2/CXR8

    FILESETLIST="images_01.tar.gz images_02.tar.gz"
    for FILESET in ${FILESETLIST}
        cp /shared/rsaas/pacole2/CXR8/${FILESET} /data/pacole2/CXR8
        cd /data/pacole2/CXR8/
        tar -xzf ${FILESET}
        cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/freq-sr/
echo "Running python script now"
python train.py --data /data/pacole2/CXR8/images/
