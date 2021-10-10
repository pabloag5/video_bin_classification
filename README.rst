VIDEO CLASSIFICATION
====================

This project aims to classify videos into two classes.
The tested architectures are:
- extended traditional 2D image classification to video frames
- 10 channel optical flow to analise motion
- LSTM to video frames
- classify using audio cue from videos
- 3D CNN (3D convolution)
- split 3D CNN

Steps to train and test:
- The modules read MP4 files arranged as `data/videos/class/` for decoding and training, and `data/test/` for evaluation.
- Videos should be decoded using `preprocessing.py`
- preparing video data with `datapreparation.py` to get train and validation datasets
- train the models with `train_model.py`; note: the script runs one model at a time, meaning if is used one architecture with different training parameters those are considered separated models
- evaluate models against test dataset with `eval_models.py`; note: the script evals one model at a time; it reads RAW MP4 files.

The modules consist of the following:

preprocessing
=============
The module contains all methods required for preprocessing the data before feeding
the models.
The raw data is splitted in video files (mp4) and text file (csv), and are stored in
different folders.
The text file maps video code and labels.

The module executes preprocessing tasks from raw data to decoding video files according
to models requirements.

Videos are decoded into folders per model per video.
From data folder containing MP4 files into `model_x/label0.0/videoid/*.png` and 
`model_x/label1.0/videoid/*.png` where `model_x` and `labelx.0` are the respective model and classes.


arguments:
---------
    - rawpreprocessing (boolean): rename, rearrange, get labels, get test dataset
    - model1frames (boolean): extract frames 1 FPS ~ 30 minutes
    - model1optflow (boolean): extract optical flow 15 FPS ~ 20 hours
    - model1specgram (boolean): extract audio spectrogram ~ 12 minutes
    - model3decoding (boolean): extract frames (30 frames per video) ~ 30 minutes

datapreparation
===============
- The scripts splits the experiment data into training and validation data sets
per model for k number of folds.
The train and validation indexes are the same for all models towards having a
fair comparison among models.
- Creates train and validation datasets per model per fold
- Creates dataloaders per model per fold

Models syntax and data:
----------------------
model1 --> cnn:
    (spatial cnn + lstm)   -     (motion cnn)     -  (spectrogram cnn)
        model1frames       -  model1opticalflows  -   model1specgram

model3 --> 3dcnn:
    (3Dcnn) | (2plus1d)
       model3frames

creativedatasets
================
Create custom dataset classes
    CreativeDataset: Dataset classes for 3 channel models:
        - model1frames
        - model1specgram
        - model3frames
    OptflowDataset: Dataset for custom channel dimension:
        - model1optflow

creatives_models
================
creatives_models contains models architectures classes.
All models are using pretrained ResNet/3DResNet models and changed last fully
connected layer to adapt to our dataset labels.
    From Pytorch models documentation:
    https://pytorch.org/docs/stable/torchvision/models.html
    Instancing a pre-trained model will download its weights to a cache directory.

Creatives models architectures:
    - FrameResnetLSTM: CNN + LSTM
    - FrameResNet: CNN
    - OptFlowResNet: custom 10-channel CNN
    - SpecgramResNet: CNN
    - ResNet3D: 3D CNN
    - ResNet2Plus1D: 3D CNN

train_model
===========

This module contains methods that load and set models architecture,
get dataloaders of decoded videos, define training parameters,
train networks, store to disk trained network parameters 
and error measurements in CSV and PNG files.

arguments
---------
    - model_type: `spt`, `sptlstm`, `mtn`, `audio`, `3dcnn`, `2plus1d`
    - dryrun: test architectures with initial weights
    - epochs: default=50
    - opt: Optimizer algorithm, default=`adam`, (`sgd`, `adam`)
    - lr: Learning rate, default=0.01
    - folds: Cross-validation folds, default=1
    - resume: Resume training from checkpoint, default=False

eval_models
===========
This module contains methods that load pretrained models architecture,
read folder containing RAW videos, and inference model against videos.

arguments
---------
    - dataset type to read: `resampled`, `undersampled`
    - model to load: `spatial`, `audio`, `3dcnn`, `2plus1d`.

- The script loads selected pretrained architecture
- Decodes test dataset and load it to dataloader
- Inferences the selected model against data
- Calculates evaluation metrics
- Creates graphical confusion matrix
- Creates ROC curves


how to use
==========
RAW videos must be arrange as `data/videos/class/`.
