"""
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
"""
import os

from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from preprocessing import get_creatives
from creativedatasets import CreativeDataset, OptflowDataset


def _split_dataset(creativespath, models, n_splits):
    """split dataset into train and validation sets for n_splits folds
    inputs:
        creativespath: path to video files
        models: list of model names
        n_splits: number of folds
    outputs:
        models_data: returns list containing dictionary per fold 
            with train and validation path per model
        labels_dict: returns serialized labels
    """
    # get creatives list
    crtvs, crtvslbls, labels = get_creatives(creativespath)
    labels_dict = {}
    for i, lbl in enumerate(labels):
        labels_dict[lbl] = i

    # get k-folds train and validation indexes
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)
    data_folds = []
    for splits in range(0, n_splits):
        train_indx, test_indx = next(sss.split(crtvs, crtvslbls))
        # get datasets
        # same train and validation split indexes for all models
        models_data = {}
        for model in models:
            data = {}
            labels = {}
            models_data[model] = {}
            data["train"] = [
                crtvs[ind].replace("parents2", model) for ind in train_indx
            ]
            labels["train"] = [crtvslbls[ind] for ind in train_indx]
            data["val"] = [crtvs[ind].replace("parents2", model) for ind in test_indx]
            labels["val"] = [crtvslbls[ind] for ind in test_indx]
            models_data[model] = {
                "path": model + "/",
                "data": data,
                "labels": labels,
            }
        data_folds.append(models_data)
    return data_folds, labels_dict


def get_dataloader(pin_memory=True, num_workers=2, n_splits=1):
    # set paths to data folders
    creativepath = "parents2/"
    datapath = "data/videos/"
    model1frames = "model1frames"
    model1optflow = "model1opticalflows"
    model1specgram = "model1specgram"
    model3frames = "model3frames"
    models = [model1frames, model1optflow, model1specgram, model3frames]
    # models = [model1frames, model1specgram, model3frames]
    # models = [model1specgram]
    # list labels and data size
    creativespath = datapath + creativepath
    # labels = os.listdir(creativespath)
    # for label in labels:
    #     print(label)
    #     print(f"Total creatives: {len(os.listdir(os.path.join(creativespath, label)))}")

    data_folds, labels_dict = _split_dataset(creativespath, models, n_splits)

    """
    ResNet
    ------
    From Pytorch image classification models documentation:
    https://pytorch.org/hub/pytorch_vision_resnet/
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are
    expected to be at least 224. The images have to be loaded in to a range of [0, 1]
    and then normalized using:
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    
    3DResNet
    --------
    From Pytorch video classification models documentation:
    https://pytorch.org/docs/stable/torchvision/models.html#video-classification
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB videos of shape (3 x T x H x W),
    where H and W are expected to be 112, and T is a number of video frames in a clip.
    The images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.43216, 0.394666, 0.37645] and std = [0.22803, 0.22145, 0.216989].
    The normalization parameters are different from the image classification ones,
    and correspond to the mean and std from Kinetics-400.
    """

    def _collate_fn_r3d(batch):
        frms_batch, lbl_batch = list(zip(*batch))
        frms_batch = [frms for frms in frms_batch if len(frms) > 0]
        lbl_batch = [
            torch.tensor(lbl)
            for lbl, frms in zip(lbl_batch, frms_batch)
            if len(frms) > 0
        ]
        frms_tensor = torch.stack(frms_batch)
        frms_tensor = torch.transpose(frms_tensor, 2, 1)
        lbls_tensor = torch.stack(lbl_batch)
        return frms_tensor, lbls_tensor

    def _collate_fn_hbrd(batch):
        frms_batch, lbl_batch = list(zip(*batch))
        frms_batch = [frms for frms in frms_batch if len(frms) > 0]
        lbl_batch = [
            torch.tensor(lbl)
            for lbl, frms in zip(lbl_batch, frms_batch)
            if len(frms) > 0
        ]
        frms_tensor = torch.stack(frms_batch)
        lbls_tensor = torch.stack(lbl_batch)
        return frms_tensor, lbls_tensor

    # train instance of creative dataset class
    dataloaders_folds = []
    for models_data in data_folds:
        models_dataloader = {}
        for key in models_data.keys():
            # print(f"\n***************** {key} *****************\n")
            if key == "model3frames":
                h, w = 112, 112
                mean = [0.43216, 0.394666, 0.37645]
                std = [0.22803, 0.22145, 0.216989]
            elif key == "model1opticalflows":
                h, w = 224, 224
                mean = [0.5]
                std = [0.5]
            else:
                h, w = 224, 224
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            # resize and augmentation for traning dataset opticalflow
            # opticalflow model requires stacked grayscale opticalflow
            if key == "model1opticalflows":
                train_transformer = transforms.Compose(
                    [
                        transforms.Resize((h, w)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
                # resize validation dataset
                val_transformer = transforms.Compose(
                    [
                        transforms.Resize((h, w)),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                # resize and augmentation for traning dataset
                train_transformer = transforms.Compose(
                    [
                        transforms.Resize((h, w)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
                # resize validation dataset
                val_transformer = transforms.Compose(
                    [
                        transforms.Resize((h, w)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            dataset = {}
            if key != "model1opticalflows":
                dataset["train"] = CreativeDataset(
                    crtvs=models_data[key]["data"]["train"],
                    labels=models_data[key]["labels"]["train"],
                    labels_dict=labels_dict,
                    transform=train_transformer,
                )
                # print("train:", len(dataset["train"]))
                # # show train sample
                # print(" ------- train sample ------- ")
                # imgs, label = dataset["train"][10]
                # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

                # validation instance of creative dataset class
                dataset["val"] = CreativeDataset(
                    crtvs=models_data[key]["data"]["val"],
                    labels=models_data[key]["labels"]["val"],
                    labels_dict=labels_dict,
                    transform=val_transformer,
                )
                # print("val:", len(dataset["val"]))
                # # show val sample
                # imgs, label = dataset["val"][5]
                # print(" ---- validation sample ---- ")
                # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

            else:
                dataset["train"] = OptflowDataset(
                    crtvs=models_data[key]["data"]["train"],
                    labels=models_data[key]["labels"]["train"],
                    labels_dict=labels_dict,
                    transform=train_transformer,
                )
                # print("train:", len(dataset["train"]))
                # # show train sample
                # print(" ------- train sample ------- ")
                # imgs, label = dataset["train"][10]
                # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

                # validation instance of creative dataset class
                dataset["val"] = OptflowDataset(
                    crtvs=models_data[key]["data"]["val"],
                    labels=models_data[key]["labels"]["val"],
                    labels_dict=labels_dict,
                    transform=val_transformer,
                )
                # print("val:", len(dataset["val"]))
                # # show val sample
                # imgs, label = dataset["val"][5]
                # print(" ---- validation sample ---- ")
                # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

            models_data[key]["dataset"] = dataset

            # training and validation dataloader
            # stack resized frames into Pytorch tensor of shape [n_frames, 3, pxl, pxl]
            #   e.i. [30, 3, 112, 112] where n_frames = 30,
            #       channels = 3, height = 112, width = 112
            #   determine batch_size during training
            #   e.i. batch_size=8 -> tensor shape [8, 30, 3, 112, 112]
            modeldataloader = {}
            batch_size = 1
            if key == "model3frames":
                modeldataloader["train"] = DataLoader(
                    models_data[key]["dataset"]["train"],
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate_fn_r3d,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                modeldataloader["val"] = DataLoader(
                    models_data[key]["dataset"]["val"],
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate_fn_r3d,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            else:
                modeldataloader["train"] = DataLoader(
                    models_data[key]["dataset"]["train"],
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate_fn_hbrd,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                modeldataloader["val"] = DataLoader(
                    models_data[key]["dataset"]["val"],
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate_fn_hbrd,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )

            models_dataloader[key] = {}
            models_dataloader[key]["dataloader"] = modeldataloader
            # print("------------- sample dataloaders --------------")
            # for xb, yb in models_dataloader[key]["dataloader"]["train"]:
            #     print(key, xb.shape, yb.shape)
            #     break
            # for xb, yb in models_dataloader[key]["dataloader"]["val"]:
            #     print(key, xb.shape, yb.shape)
            #     break
        dataloaders_folds.append(models_dataloader)

    return dataloaders_folds, labels_dict


# models_dataloader = get_dataloader()
