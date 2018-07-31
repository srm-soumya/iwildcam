#! /bin/bash

mkdir data && cd data

wget https://storage.googleapis.com/iwildcam_2018_us/train_val_sm.tar.gz
wget https://storage.googleapis.com/iwildcam_2018_us/test_sm.tar.gz
wget https://storage.googleapis.com/iwildcam_2018_us/iwildcam2018_annotations.tar.gz

tar -zxvf train_val_sm.tar.gz
tar -zxvf test_sm.tar.gz
tar -zxvf iwildcam2018_annotations.tar.gz

cd ..

python preprocess.py -d data
