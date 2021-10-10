import torch
import torch.nn as nn
from torchvision import models

"""
creatives_models
================
creatives_models contains models architectures classes.
All models are using pretrained ResNet/3DResNet models and changed last fully
connected layer to adapt to our dataset labels.
    From Pytorch models documentation:
    https://pytorch.org/docs/stable/torchvision/models.html
    Instancing a pre-trained model will download its weights to a cache directory.

reatives models architectures:
    - FrameResnetLSTM: CNN + LSTM
    - FrameResNet: CNN
    - OptFlowResNet: custom 10-channel CNN
    - SpecgramResNet: CNN
    - ResNet3D: 3D CNN
    - ResNet2Plus1D: 3D CNN
"""


class FrameResnetLSTM(nn.Module):
    """
    FrameResnetLSTM: hybrid CNN model -> ResNet + LSTM
    Custom architecture for frames and specgram models.
    Uses pretrained ResNet model as a feature extractor.
    TODO: describe network architecture
    """

    def __init__(self, params_model):
        super(FrameResnetLSTM, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        finetune = params_model["finetune"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        baseModel = models.resnet18(pretrained=True)
        if not finetune:
            for param in baseModel.parameters():
                param.requires_grad = False  # no finetune
        num_ftrs = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(
            input_size=num_ftrs, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers
        )
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)

    def forward(self, x):
        # b_z: batch_size - ts: imgs per creative - c: channels - h: height - w: width
        b_z, ts, c, h, w = x.shape
        i = 0
        y = self.baseModel((x[:, i]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for i in range(1, ts):
            y = self.baseModel((x[:, i]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))

        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FrameResNet(nn.Module):
    """FrameResNet: CNN model for frames"""

    def __init__(self, params_model):
        super(FrameResNet, self).__init__()
        num_classes = params_model["num_classes"]
        finetune = params_model["finetune"]
        # load pretrained model
        self.model = models.resnet18(pretrained=True)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False  # no finetune

        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # calculate mean prediction of all frames per video
        return torch.mean(self.model(x.squeeze(0)), dim=0, keepdim=True)


class OptFlowResNet(nn.Module):
    """
    OptFlowResNet: CNN model for optical flow.
    Accepts stacked optical flow.
    """

    def __init__(self, params_model, in_channels=10):
        super(OptFlowResNet, self).__init__()
        num_classes = params_model["num_classes"]
        # load pretrained model
        self.model = models.resnet18(pretrained=True)

        # original definition of the first layer on the resnet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # stacked optflow case
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # optical flow model requires fine-tuning since first layer is channel=10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # calculate mean prediction of all stacked optical flows per video
        return torch.mean(self.model(x.squeeze(0)), dim=0, keepdim=True)


class SpecgramResNet(nn.Module):
    """SpecgramResNet: CNN model for spectrograms"""

    def __init__(self, params_model):
        super(SpecgramResNet, self).__init__()
        num_classes = params_model["num_classes"]
        finetune = params_model["finetune"]
        # load pretrained model
        self.model = models.resnet18(pretrained=True)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False  # no finetune

        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        # remove stacked dimension since is one image per video
        return self.model(x.squeeze(0))


class ResNet3D(nn.Module):
    def __init__(self, params_model):
        super(ResNet3D, self).__init__()
        num_classes = params_model["num_classes"]
        finetune = params_model["finetune"]
        # load pretrained model
        self.model = models.video.r3d_18(pretrained=True, progress=False)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False  # no finetune

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ResNet2Plus1D(nn.Module):
    def __init__(self, params_model):
        super(ResNet2Plus1D, self).__init__()
        num_classes = params_model["num_classes"]
        finetune = params_model["finetune"]
        # load pretrained model
        self.model = models.video.r2plus1d_18(pretrained=True, progress=False)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False  # no finetune

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)
