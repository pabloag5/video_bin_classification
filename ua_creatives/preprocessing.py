"""
preprocessing
=============
This module contains all methods required for preprocessing the data before feeding
the models.
The raw data is splitted in video files (mp4) and text file (csv), and are stored in
different folders.
Raw video file name is a long text with metadata including, date, resolution and
video code. The text file maps video code and labels.
The module executes preprocessing tasks from raw data to decoding video files according
to models requirements.
arguments:
---------
    - rawpreprocessing (boolean): rename, rearrange, get labels, get test dataset
    - model1frames (boolean): extract frames 1 FPS ~ 30 minutes
    - model1optflow (boolean): extract optical flow 15 FPS ~ 20 hours
    - model1specgram (boolean): extract audio spectrogram ~ 12 minutes
    - model3decoding (boolean): extract frames (30 frames per video) ~ 30 minutes
"""
import argparse
import logging
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms.functional import to_pil_image
import wave


def rename_creative(creativepath, creativelist):
    """
    rename_creative: Rename file to creative code
    inputs:
        creativepath: path
        creativelist: creative list of files
    """
    # define regex pattern
    pattern = re.compile(r".+(ROV\w+)\.mp4")
    for creative in creativelist:
        match = pattern.findall(creative)
        if match:
            # print(creative, "video", match[0])
            os.rename(creativepath + creative, creativepath + match[0] + ".mp4")


def labels2creatives(creativepath, labelsdf):
    """
    labels2creatives: return dataframe with labels of creative's video files
    inputs:
        creativepath: path of video files
        labelsdf: dataframe with labels
    outputs:
        creatives dataframe containing creative code and label
    """
    creativelist = os.listdir(creativepath)
    # get classes labels
    classes = pd.read_csv(labelsdf)
    creatives = pd.DataFrame(creativelist, columns=["creativecode"])
    creatives["creativecode"] = creatives["creativecode"].str.split(
        pat=".mp4", expand=True
    )
    creatives = creatives.merge(classes, how="left", on=["creativecode"])
    creatives.dropna(inplace=True)
    creatives.reset_index(inplace=True)
    return creatives.drop(["index", "fcvr"], axis="columns")


def split_by_label(creativepath, datapath, labels):
    """
    split_by_label: Creates label folders if not exists and move video
    files into them.
    inputs:
        creativepath: current source path of video files
        datapath: destination path (e.g. data/creatives/)
        labels: dataframe with creative codes and labels
    """
    for i, creative in enumerate(labels["creativecode"]):
        if os.path.exists(creativepath + creative + ".mp4"):
            srccreative = creativepath + creative + ".mp4"
            labelpath = os.path.join(datapath, "label" + str(labels.loc[i, "label"]))
            descreative = os.path.join(labelpath, creative + ".mp4")
            os.makedirs(labelpath, exist_ok=True)
            shutil.move(srccreative, descreative)


def decodecreative(creative, opticalflow=False):
    """
    decodecreative: extract frames at 1 fps and optical flow from creative
    inputs:
        creative: path to creative video
        opticalflow: True if calculation of optical flow is required
    outputs:
        frames: list of video frames 1fps
        optflows: list of optical flow every two frames
        crtvlen: creative length
    """
    crtvcap = cv2.VideoCapture(creative)
    crtvlen = int(crtvcap.get(cv2.CAP_PROP_FRAME_COUNT))
    totalframes = int(crtvlen / crtvcap.get(cv2.CAP_PROP_FPS))
    totalof = int(crtvlen / 2)
    print(f"Number of frames to extract: {totalframes}")
    frames = []
    optflows = []
    framelist = np.linspace(0, crtvlen - 1, totalframes + 1, dtype=np.int16)
    oflist = np.linspace(0, crtvlen - 1, totalof, dtype=np.int16)
    if opticalflow:
        ret, frame = crtvcap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        for frm in range(crtvlen):
            ret, frame2 = crtvcap.read()
            if ret & (frm in oflist):
                nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                # magnitude and direction
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                optflows.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
                if frm in framelist:
                    frames.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                prvs = nxt
    else:
        for frm in range(crtvlen):
            ret, frame = crtvcap.read()
            if ret & (frm in framelist):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    crtvcap.release()
    return frames, optflows, crtvlen


def creative2frames(creative, n_frames=1):
    """
    creative2frames: captures video frames
    inputs:
        creative: video file path, format mp4
        n_frames: number of frames to extract from video. Two frames by default.
    outputs:
        frames: list of video frames extracted
        crtvlen: total number of frames in video
    """
    frames = []
    crtvcap = cv2.VideoCapture(creative)
    crtvlen = int(crtvcap.get(cv2.CAP_PROP_FRAME_COUNT))
    framelist = np.linspace(0, crtvlen - 1, n_frames + 1, dtype=np.int16)

    for frm in range(crtvlen):
        ret, frame = crtvcap.read()
        if ret & (frm in framelist):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    crtvcap.release()
    return frames, crtvlen


def creative2spectrogram(creative, audiopath, specgrmpath):
    """
    creative2spectrogram:
        - Extract audio cue from video into wav file
            (NOTE: requires ffmpeg library installed)
        - Decode audio file and plot spectrogram
        - save to disk spectrogram
    inputs:
        creative: path to video file
        audiopath: path to wav audio file
        specgrmpath: path to store spectrogram as jpg file
    """
    command = f"ffmpeg -y -i {creative} -ab 160k -ac 2 -ar 44100 -vn {audiopath}"
    subprocess.call(command, shell=True)
    with wave.open(audiopath, "r") as wav:
        wavframes = wav.readframes(-1)
        sound_info = np.fromstring(wavframes, dtype=np.int16)
        frame_rate = wav.getframerate()
    plt.ioff()
    fig = plt.figure(num=None, figsize=(12, 12))
    plt.specgram(sound_info, Fs=frame_rate)
    plt.axis("off")
    plt.savefig(specgrmpath, bbox_inches="tight")
    plt.close(fig)


def frames2disk(frames=None, framespath=None, optflows=None, optflowspath=None):
    """
    frames2disk: stores video frames in BGR format into framespath
    input:
        frames: video frames list
        framespath: path to store video frames
        optflows: video optical flows list
        optflowspath: path to store optical flows
    """
    if frames:
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            framepath = os.path.join(framespath, "frame" + str(i).zfill(4) + ".jpg")
            cv2.imwrite(framepath, frame)
    if optflowspath:
        for i, optflow in enumerate(optflows):
            optflowpath = os.path.join(
                optflowspath, "optflow" + str(i).zfill(4) + ".jpg"
            )
            cv2.imwrite(optflowpath, optflow)


def get_creatives(imgdatapath):
    """
    get_creatives: returns list of creatives and labels
    inputs:
        imgdatapath: root path to image data
    outputs:
        crtv_ids: list of creatives path
        labels: list of label per creative's path
        labelslist: list of labels
    """
    labelslist = sorted(os.listdir(imgdatapath))
    crtv_ids = []
    labels = []
    for label in labelslist:
        labelpath = os.path.join(imgdatapath, label)
        creativeslist = os.listdir(labelpath)
        crtvfrmpath = [
            os.path.join(labelpath, crtv.replace(".mp4", "")) for crtv in creativeslist
        ]
        crtv_ids.extend(crtvfrmpath)
        labels.extend([label] * len(creativeslist))
    return crtv_ids, labels, labelslist


def denormalize(x_, mean, std):
    """
    denormalize: denormalize image
    inputs:
        x_: image in 3-channel format
        mean: mean of channel
        std: standard deviation of channel
    outputs: returns denormalize image
    """
    x = x_.clone()
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    return to_pil_image(x)


def stratified_sample_df(df, col, frac):
    """
    stratified_sample_df: sample dataset
    inputs:
        df: dataframe with creatives code and labels
        col: dimension to sample data (e.g. label)
    returns sampled data dataframe
    """
    df_ = df.groupby(col).apply(lambda x: x.sample(frac=frac))
    df_.index = df_.index.droplevel(0)
    return df_


def test_sample(creativepath, testpath, creatives):
    """
    sample test dataset and move files to test folder
    inputs:
        creativepath: path of creatives video files
        testpath: path of test creatives video files
        creatives: dataframe with creatives and labels
    """
    testdf = stratified_sample_df(creatives, "label", 0.05)
    for i, creative in enumerate(testdf["creativecode"]):
        srccreative = creativepath + creative + ".mp4"
        descreative = os.path.join(testpath, creative + ".mp4")
        os.makedirs(testpath, exist_ok=True)
        shutil.move(srccreative, descreative)
    testdf.to_csv(testpath + "test.csv", index=False)
    print(f"Test dataset created in {testpath}")


def get_args():
    """
    get_args: get preprocessing arguments
    - rawpreprocessing: rename, rearrange, get labels, get test dataset
    - model1frames: extract frames 1 FPS ~ 30 minutes
    - model1optflow: extract optical flow 15 FPS ~ 20 hours
    - model1specgram: extract audio spectrogram ~ 12 minutes
    - model3decoding: extract frames (30 frames per video) ~ 30 minutes
    """
    parser = argparse.ArgumentParser(description="preprocessing arguments")
    parser.add_argument(
        "--rawpreprocessing", help="Run preprocessing to raw files", default=False
    )
    parser.add_argument(
        "--model1frames", help="model_1 - decode frames 1 FPS", default=False
    )
    parser.add_argument(
        "--model1optflow", help="model_1 - decode optical flow 15 FPS", default=False
    )
    parser.add_argument(
        "--model1specgram", help="model_1 - decode spectrogram", default=False
    )
    parser.add_argument(
        "--model3decoding", help="model_3 - decode frames 30 per video", default=False
    )
    args = parser.parse_args()
    return args


def main():
    ARGS = get_args()
    logging.basicConfig(filename="decoding.log", filemode="w", level=logging.DEBUG)

    # set data path for preprocessing files
    creativepath = "creatives/"
    datapath = "data/"
    creativelist = os.listdir(creativepath)

    if ARGS.rawpreprocessing:
        print("... running preprocessing raw files ... ")
        labelsdf = "creatives.csv"
        testpath = datapath + "test/"
        logging.info(f"preprocessing files in {datapath} and {testpath}")
        # rename files to creative code - creativecode.mp4
        print("... renaming files ... ")
        logging.info("... renaming files ... ")
        rename_creative(creativepath, creativelist)
        # set label to videos
        print("... setting label to creative codes ... ")
        logging.info("... setting label to creative codes ... ")
        creatives = labels2creatives(creativepath, labelsdf)
        # create test dataset
        print("... creating test dataset ... ")
        logging.info("... creating test dataset ... ")
        test_sample(creativepath, testpath, creatives)
        # split train dataset by classes
        print("... split train dataset by classes ... ")
        logging.info("... split train dataset by classes ... ")
        split_by_label(creativepath, datapath + creativepath, creatives)

    creativespath = datapath + creativepath
    labels = os.listdir(creativespath)
    for label in labels:
        print(label)
        print(f"Total creatives: {len(os.listdir(os.path.join(creativespath, label)))}")

    # model 1: decode frames at 1 FPS, (optional)optical flow at 15 FPS
    if ARGS.model1frames:
        print("... running decoding model_1 frames ... ")
        framespath = "model1frames/"
        optflowspath = "model1opticalflows/"
        logging.info(f"decoding creatives in {framespath} and {optflowspath}")
        since = time.time()
        for root, dirs, files in os.walk(creativespath, topdown=False):
            for name in files:
                crtvpath = os.path.join(root, name)
                frames, optflows, crtvlen = decodecreative(
                    crtvpath, opticalflow=ARGS.model1optflow
                )
                framepath = crtvpath.replace(creativepath, framespath)
                framepath = framepath.replace(".mp4", "")
                os.makedirs(framepath, exist_ok=True)
                optflowpath = None
                if ARGS.model1optflow:
                    optflowpath = crtvpath.replace(creativepath, optflowspath)
                    optflowpath = optflowpath.replace(".mp4", "")
                    os.makedirs(optflowpath, exist_ok=True)
                    logging.info(f"decoding of creatives in {optflowspath}")
                frames2disk(frames, framepath, optflows, optflowpath)
                logging.info(f"{crtvpath} decoded")
        time_elapsed = time.time() - since
        print(
            f"Videos decoded in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}seg"
        )
        logging.info(
            f"Videos decoded in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}seg"
        )

    # model_1: decode audio to spectrogram
    if ARGS.model1specgram:
        print("... running decoding model_1 audio ... ")
        specgramspath = "model1specgram/"
        audiopath = "model1audio/"
        logging.info(f"decoding creatives in {specgramspath}")
        creativespath = datapath + creativepath
        labels = os.listdir(creativespath)
        since = time.time()
        for root, dirs, files in os.walk(creativespath, topdown=False):
            for name in files:
                crtvpath = os.path.join(root, name)
                wavpath = crtvpath.replace(creativepath, audiopath)
                wavpath = wavpath.replace(".mp4", "")
                os.makedirs(wavpath, exist_ok=True)
                specgrampath = crtvpath.replace(creativepath, specgramspath)
                specgrampath = specgrampath.replace(".mp4", "")
                os.makedirs(specgrampath, exist_ok=True)
                creative2spectrogram(
                    crtvpath, wavpath + "/audio.wav", specgrampath + "/specgram.jpg"
                )
                print(f"{crtvpath} audio decoded")
                logging.info(f"{crtvpath} spectrogram decoded")
        time_elapsed = time.time() - since
        logging.info(
            f"Audio decoded in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}seg"
        )

    # model_3: decode creatives frames - n_frames = 30
    if ARGS.model3decoding:
        print("... running decoding model_3 frames ... ")
        framespath = "model3frames/"
        logging.info(f"decoding creatives in {framespath}")
        creativespath = datapath + creativepath
        since = time.time()
        for root, dirs, files in os.walk(creativespath, topdown=False):
            for name in files:
                crtvpath = os.path.join(root, name)
                frames, crtvlen = creative2frames(crtvpath, n_frames=30)
                framepath = crtvpath.replace(creativepath, framespath)
                framepath = framepath.replace(".mp4", "")
                os.makedirs(framepath, exist_ok=True)
                frames2disk(frames, framepath)
                print(f"{crtvpath} decoded")
                logging.info(f"{crtvpath} decoded")
        time_elapsed = time.time() - since
        logging.info(
            f"Videos decoded in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}seg"
        )


if __name__ == "__main__":
    main()


# parentpath = "data/parents/"
# labels_folders = os.listdir(parentpath)
# models = ["model1frames", "model1opticalflows", "model1specgram", "model3frames"]
# for model in models:
#     for label in labels_folders:
#         labelpath = parentpath + label + "/"
#         creativenames = os.listdir(labelpath)
#         dstfolder = labelpath.replace("parents", model)
#         print(labelpath)
#         os.makedirs(dstfolder, exist_ok=True)
#         for creative in creativenames:
#             creativefolder = creative.replace(".mp4", "")
#             source = "data/" + model + "/" + creativefolder
#             destination = dstfolder + creativefolder
#             # print(source)
#             # print(destination)
#             shutil.move(source, destination)
