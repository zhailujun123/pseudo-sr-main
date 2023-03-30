# pseudo-sr-main
CycleGAN-Transformer-SR
README:

  - To check quota and file limit for this shared directory:

        /scratch/group/mburrisgroup/check_quota.mburrisgroup.sh


  - To fix or reset permission in this shared directory
      (make all files and dirs readable/writable by the group;
       must run by the owner of files/dirs whose permssion needs to be fixed):

        /scratch/group/mburrisgroup/fix_permission.mburrisgroup.sh

# Introduction

This repo is based on [pseudo-sr](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf).

You can get the dataset from [here](https://github.com/jingyang2017/Face-and-Image-super-resolution). After unzip, put HR dataset and LR dataset in the Dataset folder "/../Dataset/LOW/LR" and "/../Dataset/HIGH/HR", respectively. Make sure the data path configured in Train.py file is the same where it is located.

The integrated Transformer model is based on Restomrer (https://github.com/swz30/Restormer)


# Usage

First, configure the yaml file which is located at `configs/faces.yaml`. Set the root folder of face dataset to `DATA.FOLDER`.

After you download the dataset, please save the highh-resolution images to /../Dataset/HIGH/HR, and save the low-resolution images to ../../Dataset/LOW/LR. Make sure the dataset path is the same to the directory configure in Train.py

Environment: Python (Pytorch)

To train:
python3 test.py configs/faces.yaml

```
CUDA_VISIBLE_DEVICES=2,3 python3 train.py configs/faces.yaml --port 12121
```

The `--port` option is only required for multi-gpu training.
You can use a number between 49152 and 65535 for the port number. 

# Reference

Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang. Restormer: Efficient Transformer for High-Resolution Image Restoration
