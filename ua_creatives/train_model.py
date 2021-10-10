"""
train_model
===========

This module contains methods that load and set models architecture,
get dataloaders of decoded videos, define training parameters,
train networks, store to disk trained network parameters 
and error measurements in CSV and PNG files.

arguments
---------
    --model_type: `spt`, `sptlstm`, `mtn`, `audio`, `3dcnn`, `2plus1d`
    --dryrun: test architectures with initial weights
    --epochs: default=50
    --opt: Optimizer algorithm, default=`adam`, (`sgd`, `adam`)
    --lr: Learning rate, default=0.01
    --folds: Cross-validation folds, default=1
    --resume: Resume training from checkpoint, default=False
"""
import argparse
import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from creatives_models import (
    FrameResnetLSTM,
    FrameResNet,
    OptFlowResNet,
    SpecgramResNet,
    ResNet3D,
    ResNet2Plus1D,
)
from datapreparation import get_dataloader


def train_val(model, params, resume=False):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    dry_run = params["dry_run"]
    lr_scheduler = params["lr_scheduler"]
    modelpath = params["modelpath"]
    bestpath = params["bestpath"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_loss = float("inf")
    best_metric = float(0)
    best_model_wts = copy.deepcopy(model.state_dict())
    begin_epoch = 0
    checkpoint = {}
    checkpoint["loss_history"] = {}
    checkpoint["metric_history"] = {}
    # if checkpoint
    if resume:
        if os.path.exists(modelpath):
            print(f"... loading checkpoint {modelpath}")
            checkpoint = torch.load(modelpath)
            model.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            begin_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            best_loss = checkpoint["best_loss"]
            loss_history["train"].extend(checkpoint["loss_history"]["train"])
            loss_history["val"].extend(checkpoint["loss_history"]["val"])
            metric_history["train"].extend(checkpoint["metric_history"]["train"])
            metric_history["val"].extend(checkpoint["metric_history"]["val"])
            best_model_wts = torch.load(bestpath)
        else:
            print(f" no checkpoint at {modelpath}")

    for epoch in range(begin_epoch, num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch {epoch}/{num_epochs-1}, current lr={current_lr}")
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, dry_run, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        # save train phase checkpoint
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["optimizer_state_dict"] = opt.state_dict()
        checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss_history"]["train"] = loss_history["train"]
        checkpoint["metric_history"]["train"] = metric_history["train"]

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, dry_run)
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     torch.save(model.state_dict(), modelpath)
        #     print("Saved best model weights!")
        if val_metric >= best_metric:
            if (val_metric > best_metric) | (val_loss < best_loss):
                best_metric = val_metric
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), bestpath)
                print("Saved best model weights!")
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        # save validation phase checkpoint
        checkpoint["best_metric"] = best_metric
        checkpoint["best_loss"] = best_loss
        checkpoint["loss_history"]["val"] = loss_history["val"]
        checkpoint["metric_history"]["val"] = metric_history["val"]
        torch.save(checkpoint, modelpath)

        lr_scheduler.step(val_loss)
        # lr_scheduler.step(val_metric)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print(
            "train loss: %.6f, dev loss: %.6f, accuracy: %.2f"
            % (train_loss, val_loss, 100 * val_metric)
        )

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def loss_epoch(model, loss_func, dataset_dl, dry_run=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm(dataset_dl):
        if opt:
            opt.zero_grad()
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss = running_loss + loss_b

        if metric_b is not None:
            running_metric = running_metric + metric_b
        if dry_run is True:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output, target)
    if opt is not None:
        # opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def plot_loss(loss_hist, metric_hist, model_type):
    num_epochs = len(loss_hist["train"])
    plt.figure()
    plt.title(f"{model_type}\nTrain-Validation Loss")
    plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig(f"outputs4/{model_type}_loss.png")
    # plt.show()

    plt.figure()
    plt.title(f"{model_type}\nTrain-Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig(f"outputs4/{model_type}_accuracy.png")
    # plt.show()


def loss_csv(loss_hist, metric_hist, model_type):
    pd.DataFrame(loss_hist).to_csv(f"outputs4/{model_type}_loss.csv")
    pd.DataFrame(metric_hist).to_csv(f"outputs4/{model_type}_metric.csv")


def get_args():
    parser = argparse.ArgumentParser(description="Creatives Classification")
    parser.add_argument(
        "--model_type",
        "-m",
        default="audio",
        choices=["spt", "sptlstm", "mtn", "audio", "3dcnn", "2plus1d",],
        required=True,
        help="Model_type to train",
    )
    parser.add_argument(
        "--dryrun", default=False, help="test architectures with initial weights",
    )
    parser.add_argument("--epochs", default=50, help="Number of epochs")
    parser.add_argument(
        "--opt", default="adam", choices=["sgd", "adam"], help="Optimizer algorithm"
    )
    parser.add_argument("--lr", default=0.01, help="Learning rate")
    parser.add_argument("--folds", default=1, help="Cross-validation folds")
    parser.add_argument(
        "--resume", default=False, help="Resume training from checkpoint"
    )
    args = parser.parse_args()
    return args


def dry_run():
    # test architectures with initial weights
    # dry-run models
    print("************************************")
    print("******     dry-run models    *******")
    print("************************************")
    params_model = {
        "num_classes": num_classes,
        "dr_rate": 0.5,
        "finetune": False,
        "rnn_num_layers": 2,
        "rnn_hidden_size": 1024,
    }
    with torch.no_grad():
        model = FrameResnetLSTM(params_model)
        input = torch.randn((1, 44, 3, 224, 224))
        output = model(input)
        print(f"dry-run FrameResnetLSTM output: {output}")
        print(output.shape)
        print(output)

        output = []
        model = FrameResNet(params_model)
        input = torch.randn((1, 44, 3, 224, 224))
        output = model(input)
        print(f"dry-run FrameResNet output: {output}")
        print(output.shape)
        print(output)

        model = OptFlowResNet(params_model)
        input = torch.randn((1, 44, 10, 224, 224))
        output = model(input)
        print(f"dry-run OptFlowResNet output: {output}")
        print(output.shape)
        print(output)

        model = SpecgramResNet(params_model)
        input = torch.randn((1, 1, 3, 224, 224))
        output = model(input)
        print(f"dry-run SpecgramResNet output: {output}")
        print(output.shape)
        print(output)

        model = ResNet3D(params_model)
        input = torch.randn((1, 3, 30, 112, 112))
        output = model(input)
        print(f"dry-run ResNet3D output: {output}")
        print(output.shape)
        print(output)

        model = ResNet2Plus1D(params_model)
        input = torch.randn((1, 3, 30, 112, 112))
        output = model(input)
        print(f"dry-run ResNet2Plus1D output: {output}")
        print(output.shape)
        print(output)


args = get_args()
# get dataloaders
dataloaders_folds, labels_dict = get_dataloader(n_splits=int(args.folds))
num_classes = len(labels_dict)
# get model_type to train
epochs = int(args.epochs)
lr = float(args.lr)
optimizer = args.opt
resume = False
if args.resume:
    resume = True
print(f"resume {resume}")

dryrun = False
if args.dryrun:
    dryrun = True
    dry_run()

# ##################################
# setting model architecture       #
# ##################################
# determine architecture parameters
if args.model_type in ["sptlstm"]:
    params_model = {
        "num_classes": num_classes,
        "dr_rate": 0.5,
        "finetune": True,
        "rnn_num_layers": 2,
        "rnn_hidden_size": 1024,
    }
else:
    params_model = {
        "num_classes": num_classes,
        "finetune": True,
    }
for i, models_datasets in enumerate(dataloaders_folds):
    # set architecture and train and validation dataloader
    if args.model_type == "spt":
        model = FrameResNet(params_model)
        train_frms = models_datasets["model1frames"]["dataloader"]["train"]
        val_frms = models_datasets["model1frames"]["dataloader"]["val"]
    if args.model_type == "sptlstm":
        model = FrameResnetLSTM(params_model)
        train_frms = models_datasets["model1frames"]["dataloader"]["train"]
        val_frms = models_datasets["model1frames"]["dataloader"]["val"]
    if args.model_type == "mtn":
        model = OptFlowResNet(params_model)
        train_frms = models_datasets["model1opticalflows"]["dataloader"]["train"]
        val_frms = models_datasets["model1opticalflows"]["dataloader"]["val"]
    if args.model_type == "audio":
        model = SpecgramResNet(params_model)
        train_frms = models_datasets["model1specgram"]["dataloader"]["train"]
        val_frms = models_datasets["model1specgram"]["dataloader"]["val"]
    if args.model_type == "3dcnn":
        model = ResNet3D(params_model)
        train_frms = models_datasets["model3frames"]["dataloader"]["train"]
        val_frms = models_datasets["model3frames"]["dataloader"]["val"]
    if args.model_type == "2plus1d":
        model = ResNet2Plus1D(params_model)
        train_frms = models_datasets["model3frames"]["dataloader"]["train"]
        val_frms = models_datasets["model3frames"]["dataloader"]["val"]

    # initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)
    # modelpath = f"models/{model_type}_weights.pt"
    # os.makedirs("models/", exist_ok=True)
    # torch.save(model.state_dict(), modelpath)
    momentum = 0.9
    # ###################################################
    # Define model parameters                           #
    # ###################################################
    # define loss function (criterion) and optimizer
    # add weight parameter to loss function -> inbalanced classes
    # weight_imbalance = torch.tensor([1.0, 30.0])
    # The input is expected to contain raw, unnormalized scores for each class.
    # loss_func = nn.CrossEntropyLoss(weight=weight_imbalance, reduction="none").to(device)
    loss_func = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=30, verbose=1
    )
    # lr_scheduler = StepLR(opt, step_size=50, gamma=0.5)

    # set folders to save model outputs
    os.makedirs("models4/", exist_ok=True)
    os.makedirs("outputs4/", exist_ok=True)
    # set parameter dict
    model_type = args.model_type + "_" + optimizer + "_" + str(lr) + "_fold_" + str(i)
    params_train = {
        "num_epochs": epochs,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_frms,
        "val_dl": val_frms,
        "dry_run": dryrun,
        "lr_scheduler": lr_scheduler,
        "modelpath": f"models4/{model_type}_checkpoint.pt",
        "bestpath": f"models4/{model_type}_weights.pt",
    }

    # ############################################
    # train model                                #
    # ############################################
    print("\n-----------------------------------------------------\n")
    print("\n*****************************************")
    print(f" ... start training {model_type} ...")
    print("*****************************************\n")
    logging.basicConfig(
        filename=f"train_model_{model_type}.log", filemode="w", level=logging.DEBUG
    )
    start = time.time()
    model, loss_hist, metric_hist = train_val(model, params_train, resume)
    end = time.time() - start
    print("TRAINING COMPLETE")
    print(f"Fit {model_type} in {end // 60:.0f}min {end % 60:.0f}seg")
    logging.info("TRAINING COMPLETE")
    logging.info(f"Fit {model_type} in {end // 60:.0f}min {end % 60:.0f}seg")
    plot_loss(loss_hist, metric_hist, model_type)
    loss_csv(loss_hist, metric_hist, model_type)
