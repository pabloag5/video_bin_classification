"""
creativedatasets
================
Create custom dataset classes
    CreativeDataset: Dataset classes for 3 channel models:
        - model1frames
        - model1specgram
        - model3frames
    OptflowDataset: Dataset for custom channel dimension:
        - model1optflow
"""
import glob
from PIL import Image
import random

import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)


class CreativeDataset(Dataset):
    """
    Frames dataset
    ==============
    Stack into Pytorch tensor of shape [n_frames, channels, pxl, pxl]
        e.i. [30, 3, 224, 224] where n_frames = 30 frames per video,
        channels = 3 (RGB),
        height = 224, width = 224
    """

    def __init__(self, crtvs, labels, labels_dict, transform):
        """
        crtvs: List with path to creatives
        labels: List with labels of creatives
        labels_dict: labels dictionary
        transform: transform to be applied on a sample
        """
        self.transform = transform
        self.crtvs = crtvs
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.crtvs)

    def __getitem__(self, idx):
        framespath = glob.glob(self.crtvs[idx] + "/*.jpg")
        framespath = list(sorted(framespath))
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for framepath in framespath:
            frame = Image.open(framepath)
            frames.append(frame)

        seed = np.random.randint(1e9)
        framestrain = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            framestrain.append(self.transform(frame))
        if len(framestrain) > 0:
            framestrain = torch.stack(framestrain)

        return framestrain, label


class OptflowDataset(Dataset):
    """
    Optical flow dataset
    ====================
    Dataset has chunks of stacked optical flow of 10 frames.
    If last chunk is less than 10 frames is removed from sample.
    Stack into Pytorch tensor of shape [n_stacked, chunk_size, pxl, pxl]
        e.i. [40, 10, 224, 224] where n_stacked = 40 stacked optical flows,
        chunk_size = 10 (number of optical flows per stack),
        height = 224, width = 224
    """

    def __init__(self, crtvs, labels, labels_dict, transform):
        self.transform = transform
        self.crtvs = crtvs
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.crtvs)

    def __getitem__(self, idx):
        framespath = glob.glob(self.crtvs[idx] + "/*.jpg")
        framespath = list(sorted(framespath))
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for framepath in framespath:
            frame = Image.open(framepath)
            frames.append(frame)

        seed = np.random.randint(1e9)
        chunk_size = 10
        framestrain = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            framestrain.append(self.transform(frame))
        if len(framestrain) > 0:
            framestrain = torch.stack(framestrain)
            framestrain = torch.transpose(framestrain, 0, 1)
            framestrain = torch.split(framestrain, chunk_size, 1)
            if framestrain[-1].shape[1] < chunk_size:
                framestrain = framestrain[:-1]
            framestrain = torch.stack(framestrain)
            framestrain = framestrain.squeeze(1)
        return framestrain, label
