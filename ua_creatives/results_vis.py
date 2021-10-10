import matplotlib.pyplot as plt
import pandas as pd

"""
import models results.
Two models:
    - resample
    - undersample
"""
resamplepath = "outputs/resample/"
undersamplepath = "outputs/undersample/"
accuracy = "_adam_0.01_metric.csv"
loss = "_adam_0.01_loss.csv"

# resample
r_spt = pd.read_csv(resamplepath + "spt" + accuracy, index_col=0)
r_sptlstm = pd.read_csv(resamplepath + "sptlstm" + accuracy, index_col=0)
r_audio = pd.read_csv(resamplepath + "audio" + accuracy, index_col=0)
r_3dcnn = pd.read_csv(resamplepath + "3dcnn" + accuracy, index_col=0)
r_2plus1d = pd.read_csv(resamplepath + "2plus1d" + accuracy, index_col=0)
# r_mtn = pd.read_csv(resamplepath + "mtn" + accuracy, index_col = 0)
r_spt = r_spt.merge(
    pd.read_csv(resamplepath + "spt" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
r_sptlstm = r_sptlstm.merge(
    pd.read_csv(resamplepath + "sptlstm" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
r_audio = r_audio.merge(
    pd.read_csv(resamplepath + "audio" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
r_3dcnn = r_3dcnn.merge(
    pd.read_csv(resamplepath + "3dcnn" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
r_2plus1d = r_2plus1d.merge(
    pd.read_csv(resamplepath + "2plus1d" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
# r_mtn = r_mtn.merge(
#     pd.read_csv(resamplepath + "mtn" + loss, index_col=0),
#     left_index=True,
#     right_index=True,
#     suffixes=("_acc", "_loss"),
# )

# bar chart compare acc
r_spt_summary = r_spt.groupby("val_acc").min("val_loss")
r_sptlstm_summary = r_sptlstm.groupby("val_acc").min("val_loss")
r_audio_summary = r_audio.groupby("val_acc").min("val_loss")
r_3dcnn_summary = r_3dcnn.groupby("val_acc").min("val_loss")
r_2plus1d_summary = r_2plus1d.groupby("val_acc").min("val_loss")
# mtn_summary = r_mtn.groupby("val_acc").min("val_loss")

r_spt_summary.reset_index(inplace=True)
r_sptlstm_summary.reset_index(inplace=True)
r_audio_summary.reset_index(inplace=True)
r_3dcnn_summary.reset_index(inplace=True)
r_2plus1d_summary.reset_index(inplace=True)
# mtn_summary.reset_index(inplace=True)

# plots
plt.style.use("seaborn-white")

# resampled
fig, axes = plt.subplots(2, 3, sharey=True)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fig.suptitle("Accuracy - loss", fontsize=14)
r_spt_summary["val_acc"].plot.bar(
    ax=axes[0][0], edgecolor="red", fill=False, label="Acc", legend=True, ylim=(0, 1),
)
r_spt_summary["val_loss"].plot(
    ax=axes[0][0], secondary_y=True, style="k--", label="loss", legend=True
)
axes[0][0].set_ylabel("Accuracy")
axes[0][0].set_title("Spatial")
axes[0][0].right_ax.set_ylim(0, 10)
axes[0][0].xaxis.set_major_formatter(plt.NullFormatter())
axes[0][0].right_ax.set_yticklabels([])
# axes[0][0].right_ax.yaxis.set_major_formatter(plt.NullFormatter())

r_sptlstm_summary["val_acc"].plot.bar(
    ax=axes[0][1], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
r_sptlstm_summary["val_loss"].plot(
    ax=axes[0][1], secondary_y=True, style="k--", label="loss", legend=False
)
axes[0][1].set_title("Spatial-LSTM")
axes[0][1].right_ax.set_ylim(0, 10)
axes[0][1].xaxis.set_major_formatter(plt.NullFormatter())
# axes[0][1].right_ax.yaxis.set_major_formatter(plt.NullFormatter())
axes[0][1].right_ax.set_yticklabels([])

r_audio_summary["val_acc"].plot.bar(
    ax=axes[0][2], edgecolor="red", fill=False, label="Acc", ylim=(0, 1)
)
r_audio_summary["val_loss"].plot(
    ax=axes[0][2], secondary_y=True, style="k--", label="loss", legend=False
)
axes[0][2].right_ax.set_ylabel("Loss")
axes[0][2].set_title("Audio")
axes[0][2].right_ax.set_ylim(0, 10)
axes[0][2].xaxis.set_major_formatter(plt.NullFormatter())
# axes[0][2].right_ax.set_yticklabels([])


r_3dcnn_summary["val_acc"].plot.bar(
    ax=axes[1][0], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
r_3dcnn_summary["val_loss"].plot(
    ax=axes[1][0], secondary_y=True, style="k--", label="loss", legend=False
)
axes[1][0].set_ylabel("Accuracy")
axes[1][0].set_title("3DCNN")
axes[1][0].right_ax.set_ylim(0, 10)
axes[1][0].xaxis.set_major_formatter(plt.NullFormatter())
# axes[1][0].right_ax.yaxis.set_major_formatter(plt.NullFormatter())
axes[1][0].right_ax.set_yticklabels([])

r_2plus1d_summary["val_acc"].plot.bar(
    ax=axes[1][1], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
r_2plus1d_summary["val_loss"].plot(
    ax=axes[1][1], secondary_y=True, style="k--", label="loss", legend=False
)
axes[1][1].set_title("2plus1D")
axes[1][1].right_ax.set_ylim(0, 10)
axes[1][1].xaxis.set_major_formatter(plt.NullFormatter())
axes[1][1].right_ax.set_yticklabels([])

# r_mtn_summary["val_acc"].plot.bar(
#     ax=axes[1][2], edgecolor="red", fill=False, label="Acc", ylim=(0, 1)
# )
# r_mtn_summary["val_loss"].plot(
#     ax=axes[1][2], secondary_y=True, style="k--", label="loss", legend=False
# )
# axes[1][2].right_ax.set_ylabel("Loss")
axes[1][2].set_title("Motion")
# axes[1][2].right_ax.set_ylim(0, 10)
axes[1][2].xaxis.set_major_formatter(plt.NullFormatter())
# axes[1][2].right_ax.set_yticklabels([])


# undersample
u_spt = pd.read_csv(undersamplepath + "spt" + accuracy, index_col=0)
u_sptlstm = pd.read_csv(undersamplepath + "sptlstm" + accuracy, index_col=0)
u_audio = pd.read_csv(undersamplepath + "audio" + accuracy, index_col=0)
u_3dcnn = pd.read_csv(undersamplepath + "3dcnn" + accuracy, index_col=0)
u_2plus1d = pd.read_csv(undersamplepath + "2plus1d" + accuracy, index_col=0)
u_mtn = pd.read_csv(undersamplepath + "mtn" + accuracy, index_col=0)

u_spt = u_spt.merge(
    pd.read_csv(undersamplepath + "spt" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
u_sptlstm = u_sptlstm.merge(
    pd.read_csv(undersamplepath + "sptlstm" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
u_audio = u_audio.merge(
    pd.read_csv(undersamplepath + "audio" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
u_3dcnn = u_3dcnn.merge(
    pd.read_csv(undersamplepath + "3dcnn" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
u_2plus1d = u_2plus1d.merge(
    pd.read_csv(undersamplepath + "2plus1d" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)
u_mtn = u_mtn.merge(
    pd.read_csv(undersamplepath + "mtn" + loss, index_col=0),
    left_index=True,
    right_index=True,
    suffixes=("_acc", "_loss"),
)

# bar chart compare acc
u_spt_summary = u_spt.groupby("val_acc").min("val_loss")
u_sptlstm_summary = u_sptlstm.groupby("val_acc").min("val_loss")
u_audio_summary = u_audio.groupby("val_acc").min("val_loss")
u_3dcnn_summary = u_3dcnn.groupby("val_acc").min("val_loss")
u_2plus1d_summary = u_2plus1d.groupby("val_acc").min("val_loss")
u_mtn_summary = u_mtn.groupby("val_acc").min("val_loss")

u_spt_summary.reset_index(inplace=True)
u_sptlstm_summary.reset_index(inplace=True)
u_audio_summary.reset_index(inplace=True)
u_3dcnn_summary.reset_index(inplace=True)
u_2plus1d_summary.reset_index(inplace=True)
u_mtn_summary.reset_index(inplace=True)


# resampled
fig, axes = plt.subplots(2, 3, sharey=True)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fig.suptitle("Accuracy - loss", fontsize=14)
u_spt_summary["val_acc"].plot.bar(
    ax=axes[0][0], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
u_spt_summary["val_loss"].plot(
    ax=axes[0][0], secondary_y=True, style="k--", label="loss"
)
axes[0][0].set_ylabel("Accuracy")
axes[0][0].set_title("Spatial")
axes[0][0].right_ax.set_ylim(0, 1)
axes[0][0].xaxis.set_major_formatter(plt.NullFormatter())
# axes[0][0].right_ax.yaxis.set_major_formatter(plt.NullFormatter())
axes[0][0].right_ax.set_yticklabels([])

u_sptlstm_summary["val_acc"].plot.bar(
    ax=axes[0][1], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
u_sptlstm_summary["val_loss"].plot(
    ax=axes[0][1], secondary_y=True, style="k--", label="loss", legend=False
)
axes[0][1].set_title("Spatial-LSTM")
axes[0][1].right_ax.set_ylim(0, 1)
axes[0][1].xaxis.set_major_formatter(plt.NullFormatter())
# axes[0][1].right_ax.yaxis.set_major_formatter(plt.NullFormatter())
axes[0][1].right_ax.set_yticklabels([])

u_audio_summary["val_acc"].plot.bar(
    ax=axes[0][2], edgecolor="red", fill=False, label="Acc", ylim=(0, 1)
)
u_audio_summary["val_loss"].plot(
    ax=axes[0][2], secondary_y=True, style="k--", label="loss", legend=False
)
axes[0][2].right_ax.set_ylabel("Loss")
axes[0][2].set_title("Audio")
axes[0][2].right_ax.set_ylim(0, 1)
axes[0][2].xaxis.set_major_formatter(plt.NullFormatter())
# axes[0][2].right_ax.set_yticklabels([])

u_3dcnn_summary["val_acc"].plot.bar(
    ax=axes[1][0], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
u_3dcnn_summary["val_loss"].plot(
    ax=axes[1][0], secondary_y=True, style="k--", label="loss", legend=False
)
axes[1][0].set_ylabel("Accuracy")
axes[1][0].set_title("3DCNN")
axes[1][0].right_ax.set_ylim(0, 1)
axes[1][0].xaxis.set_major_formatter(plt.NullFormatter())
# axes[1][0].right_ax.yaxis.set_major_formatter(plt.NullFormatter())
axes[1][0].right_ax.set_yticklabels([])

u_2plus1d_summary["val_acc"].plot.bar(
    ax=axes[1][1], edgecolor="red", fill=False, label="Acc", ylim=(0, 1),
)
u_2plus1d_summary["val_loss"].plot(
    ax=axes[1][1], secondary_y=True, style="k--", label="loss", legend=False
)
axes[1][1].set_title("2plus1D")
axes[1][1].right_ax.set_ylim(0, 1)
axes[1][1].xaxis.set_major_formatter(plt.NullFormatter())
axes[1][1].right_ax.set_yticklabels([])

u_mtn_summary["val_acc"].plot.bar(
    ax=axes[1][2], edgecolor="red", fill=False, label="Acc", ylim=(0, 1)
)
u_mtn_summary["val_loss"].plot(
    ax=axes[1][2], secondary_y=True, style="k--", label="loss", legend=False
)
axes[1][2].right_ax.set_ylabel("Loss")
axes[1][2].set_title("Motion")
axes[1][2].right_ax.set_ylim(0, 1)
axes[1][2].xaxis.set_major_formatter(plt.NullFormatter())
# axes[1][2].right_ax.set_yticklabels([])


"""
2plus1d 376 min 47 seg
spt 951min 57g
audio 48 min 15 seg

3dcnn 263 min 34 seg
spt 181 min 9 seg
sptlstm 188min 28 seg
audio 14min 48 seg
2plus1d 187 min 34 seg
mtn 
"""

outputs3 = "outputs/outputs3/"
outputs4 = "outputs/outputs4/"

models = ["2plus1d", "3dcnn", "audio", "spt"]
for output in [outputs3, outputs4]:
    for m, model in enumerate(models):
        for fold in range(5):
            for i, metric in enumerate(["loss", "metric"]):
                file = f"{output}{model}_adam_0.01_fold_{fold}_{metric}.csv"
                if i < 1:
                    df = pd.read_csv(
                        file,
                        header=0,
                        names=["epoch", f"train_{metric}", f"val_{metric}"],
                        index_col=0,
                    )
                else:
                    df = df.merge(
                        pd.read_csv(
                            file,
                            header=0,
                            names=["epoch", f"train_{metric}", f"val_{metric}"],
                            index_col=0,
                        ),
                        left_index=True,
                        right_index=True,
                    )
                df.reset_index(inplace=True)
                df["fold"] = fold
                df["model"] = model
            if (fold == 0) & (m == 0):
                dffolds = df
            else:
                dffolds = dffolds.append(df, ignore_index=False)

    dffolds.shape
    dffolds.drop(labels="index", axis=1, inplace=True)
    dffolds.head()
    dffolds.to_csv(f"{output}outputs.csv", index=False)
