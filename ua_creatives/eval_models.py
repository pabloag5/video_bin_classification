"""
eval_models
===========
This module contains methods that load pretrained models architecture,
read folder containing RAW videos, and inference model against videos.

arguments
---------
dataset type to read: resampled, undersampled
model to load: spatial, audio, 3dcnn, 2plus1d.

- The script loads selected pretrained architecture
- Decodes test dataset and load it to dataloader
- Inferences the selected model against data
- Calculates evaluation metrics
- Creates graphical confusion matrix
- Creates ROC curves
"""


import argparse
import glob
import itertools
import os
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
import torchvision.transforms as transforms

from creatives_models import (
    FrameResNet,
    OptFlowResNet,
    SpecgramResNet,
    ResNet3D,
    ResNet2Plus1D,
)
from preprocessing import (
    decodecreative,
    creative2frames,
    creative2spectrogram,
)


def get_creative_model(model_type, num_classes=2):
    # instantiate model to eval
    params_model = {
        "num_classes": num_classes,
        "finetune": True,
    }
    if model_type == "spt":
        model = FrameResNet(params_model)
    if model_type == "mtn":
        model = OptFlowResNet(params_model)
    if model_type == "audio":
        model = SpecgramResNet(params_model)
    if model_type == "3dcnn":
        model = ResNet3D(params_model)
    if model_type == "2plus1d":
        model = ResNet2Plus1D(params_model)
    return model


def get_model_data(model_type, creative):
    """
    spt: decode frames at 1 FPS, (optional)optical flow at 15 FPS

    """
    print("... running preprocessing raw file ... ")
    if model_type == "spt":
        deco_creative, _, _ = decodecreative(creative)
    if model_type == "mtn":
        _, deco_creative, _ = decodecreative(creative, opticalflow=True)
    if model_type == "audio":
        root = creative[0:-16]
        creative2spectrogram(creative, root + "/audio.wav", root + "/specgram.jpg")
        deco_creative = Image.open(root + "/specgram.jpg")
        os.remove(root + "/audio.wav")
        os.remove(root + "/specgram.jpg")
    if model_type == "3dcnn":
        deco_creative, _ = creative2frames(creative, n_frames=30)
    if model_type == "2plus1d":
        deco_creative, _ = creative2frames(creative, n_frames=30)

    if (model_type == "3dcnn") | (model_type == "2plus1d"):
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        eval_transformer = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif model_type == "mtn":
        h, w = 224, 224
        mean = [0.5]
        std = [0.5]
        eval_transformer = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        eval_transformer = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dt_transformed = []
    if model_type == "audio":
        dt_transformed.append(eval_transformer(deco_creative))
        data_tensor = torch.stack(dt_transformed)
    if model_type == "spt":
        for frame in deco_creative:
            frame = Image.fromarray(frame)
            dt_transformed.append(eval_transformer(frame))
        data_tensor = torch.stack(dt_transformed)
    if model_type == "mtn":
        chunk_size = 10
        for frame in deco_creative:
            frame = Image.fromarray(frame)
            dt_transformed.append(eval_transformer(frame))
        data_tensor = torch.stack(dt_transformed)
        data_tensor = torch.transpose(data_tensor, 0, 1)
        data_tensor = torch.split(data_tensor, chunk_size, 1)
        if data_tensor[-1].shape[1] < chunk_size:
            data_tensor = data_tensor[:-1]
        data_tensor = torch.stack(data_tensor)
        data_tensor = data_tensor.squeeze(1)
    if (model_type == "3dcnn") | (model_type == "2plus1d"):
        for frame in deco_creative:
            frame = Image.fromarray(frame)
            dt_transformed.append(eval_transformer(frame))
        data_tensor = torch.stack(dt_transformed)
        data_tensor = torch.transpose(data_tensor, 1, 0)

    return data_tensor.unsqueeze(0)


def get_args():
    parser = argparse.ArgumentParser(description="Creatives Classification")
    parser.add_argument(
        "--dataset",
        default="u",
        choices=["r", "u"],
        help="r: resampled - u: undersampled",
    )
    parser.add_argument(
        "--model_type",
        "-m",
        default="audio",
        choices=["spt", "mtn", "audio", "3dcnn", "2plus1d"],
        required=True,
        help="Model_type to eval",
    )
    # args = parser.parse_args(["--dataset", "u", "--model_type", "2plus1d"])
    args = parser.parse_args()
    return args


# r_2plus1d_fold_0, r_3dcnn_fold_2, u_audio_fold_3, u_2plus1d_fold_0
args = get_args()
model = get_creative_model(args.model_type, num_classes=2)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.dataset == "r":
    dataset = "models3"
else:
    dataset = "models4"
modelpath = f"models/{dataset}/{args.model_type}_adam_0.01_fold_0_weights.pt"
# load pretrained model weights
model.load_state_dict(torch.load(modelpath, map_location=device))
model.to(device)

# ******************************************
#  M O D E L           E V A L U A T I O N
# ******************************************
creativepath = "data/test/demo/*.mp4"
creativelist = glob.glob(creativepath)
# reallabels = pd.read_csv(creativepath[0:-5] + "test_labels.csv")
predlabels = {}
predlabels["creativecode"] = []
predlabels["pred"] = []
print("\n*****************************************")
print(f" ... MODEL {dataset}-{args.model_type} EVALUATION ...")
print("*****************************************\n")
for creative in creativelist:
    data_tensor = get_model_data(args.model_type, creative)
    with torch.no_grad():
        out = model(data_tensor.to(device)).cpu()
        print(out.shape)
        pred = torch.argmax(out).item()
        print(f"Creative {creative[-15:]} prediction: label {pred}")
    predlabels["creativecode"].append(creative[-15:])
    predlabels["pred"].append(pred)
eval_df = reallabels.merge(pd.DataFrame(predlabels), on=["creativecode"])

print(sum(eval_df["label"].astype("int") == eval_df["pred"]) / len(eval_df))

if args.dataset == "r":
    if args.model_type == "spt":
        r_spt_df = eval_df
    if args.model_type == "audio":
        r_audio_df = eval_df
    if args.model_type == "3dcnn":
        r_3dcnn_df = eval_df
    if args.model_type == "2plus1d":
        r_2plus1d_df = eval_df
else:
    if args.model_type == "spt":
        u_spt_df = eval_df
    if args.model_type == "audio":
        u_audio_df = eval_df
    if args.model_type == "3dcnn":
        u_3dcnn_df = eval_df
    if args.model_type == "2plus1d":
        u_2plus1d_df = eval_df
    if args.model_type == "mtn":
        u_mtn_df = eval_df

models = [
    # r_spt_df,
    # r_audio_df,
    r_3dcnn_df,
    r_2plus1d_df,
    # u_spt_df,
    u_audio_df,
    # u_3dcnn_df,
    u_2plus1d_df,
    # u_mtn_df,
]
acc = []
precision = []
sensitivity = []
specificity = []
for m in models:
    fltrpredpos = m["pred"] == 1
    fltrsens = m["label"].astype("int") == 1
    fltrspec = m["label"].astype("int") == 0
    acc.append(sum(m["label"].astype("int") == m["pred"]) / len(m))
    precision.append(
        sum(m.loc[fltrpredpos, "label"].astype("int") == m.loc[fltrpredpos, "pred"])
        / len(m.loc[fltrpredpos])
    )
    sensitivity.append(
        sum(m.loc[fltrsens, "pred"] == m.loc[fltrsens, "label"].astype("int"))
        / len(m.loc[fltrsens])
    )
    specificity.append(
        sum(m.loc[fltrspec, "pred"] == m.loc[fltrspec, "label"].astype("int"))
        / len(m.loc[fltrspec])
    )
    acc, precision, sensitivity, specificity
model_types = [
    # "r_spt",
    # "r_audio",
    "r_3dcnn",
    "r_2plus1d",
    # "u_spt",
    "u_audio",
    # "u_3dcnn",
    "u_2plus1d",
    # "u_mtn",
]
labels = ["label-0", "label-1"]

# Confusion matrix best model
y_true = r_2plus1d_df["label"]
y_pred = r_2plus1d_df["pred"]
cf_matrix = confusion_matrix(y_true, y_pred)

accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
misclass = 1 - accuracy
cmap = plt.get_cmap("OrRd")
plt.figure()
plt.imshow(cf_matrix, interpolation="nearest", cmap=cmap)
plt.title("Undersampled 2plus1d - Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=0)
plt.yticks(tick_marks, labels)
thresh = cf_matrix.max() / 2
for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
    plt.text(
        j,
        i,
        "{:,}".format(cf_matrix[i, j]),
        horizontalalignment="center",
        color="white" if cf_matrix[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel(
    "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
)
plt.show()

cf_matrix
specificity = cf_matrix[0, 0] / np.sum(cf_matrix[0, :])
sensitivity = cf_matrix[1, 1] / np.sum(cf_matrix[1, :])
print(f"sensitivity: {sensitivity}")
print(f"specificity: {specificity}")

# ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="black", label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate Class 1")
plt.ylabel("True Positive Rate Class 1")
plt.title("Resampled vs. Undersampled 2plus1d - ROC")
plt.legend(loc="lower right")
plt.show()


# ROC 2plus1d
y_true_r = r_2plus1d_df["label"]
y_pred_r = r_2plus1d_df["pred"]
y_true_u = u_2plus1d_df["label"]
y_pred_u = u_2plus1d_df["pred"]
fpr_r, tpr_r, thresholds = roc_curve(y_true_r, y_pred_r)
fpr_u, tpr_u, thresholds = roc_curve(y_true_u, y_pred_u)
roc_auc_r = auc(fpr_r, tpr_r)
roc_auc_u = auc(fpr_u, tpr_u)
plt.figure()
plt.plot(fpr_r, tpr_r, color="red", label="ROC curve (area = %0.2f)" % roc_auc_r)
plt.plot(fpr_u, tpr_u, color="black", label="ROC curve (area = %0.2f)" % roc_auc_u)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate Class 1")
plt.ylabel("True Positive Rate Class 1")
plt.title("Resampled vs. Undersampled 2plus1d - ROC")
plt.legend(loc="lower right")
plt.show()
