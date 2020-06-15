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

if [[ ! -d "/data/pacole2/DeepLesion/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    mkdir /data/pacole2/DeepLesion/

    FILESETLIST="Images_png_55.zip Images_png_54.zip"
    for FILESET in ${FILESETLIST}
    do
        echo "Copying over ${FILESET}.."
        cp /shared/rsaas/pacole2/DeepLesion/${FILESET} /data/pacole2/DeepLesion/${FILESET}
        cd /data/pacole2/DeepLesion/
        unzip -qq ${FILESET}
        rm ${FILESET}
        cd /home/pacole2/
    done
fi

# Preprocess data
cd ~/Projects/freq-sr/tools/
echo "Preprocessing data..."
python process_deep_lesion.py /data/pacole2/DeepLesion/ --image_size 256 256

# Move data
cd /data/pacole2/DeepLesion/
mkdir DeepLesionPreprocessed
mv miniStudies DeepLesionPreprocessed
mv noiseStudies DeepLesionPreprocessed

echo "Zipping data..."
zip -rq DeepLesionPreprocessed.zip DeepLesionPreprocessed
mv DeepLesionPreprocessed.zip /shared/rsaas/pacole2/
