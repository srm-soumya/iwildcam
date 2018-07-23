# Steps to run the model

1. Download the datasets [train_val](https://storage.googleapis.com/iwildcam_2018_us/train_val_sm.tar.gz), [test](https://storage.googleapis.com/iwildcam_2018_us/test_sm.tar.gz), [annotations](https://storage.googleapis.com/iwildcam_2018_us/iwildcam2018_annotations.tar.gz) and store inside the data folder. 
2. Run notebook `01_18-07-2018_srm_iwildcam-analyse-dataset.ipynb`
3. Run notebook `02_18-07-2018_srm_iwildcam-split-data.ipynb`
4. `python resize.py -d data`
5. `python train.py -d data -n num_epochs`