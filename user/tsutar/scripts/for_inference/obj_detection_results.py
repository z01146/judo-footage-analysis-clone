#!/usr/bin/python3.10
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

runs_dir = "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scripts/runs"
df = pd.read_csv(runs_dir + "/detect/train2/results.csv", skipinitialspace=True)
df.set_index("epoch", inplace=True)

df = df[["val/box_loss", "val/cls_loss", "metrics/precision(B)", "metrics/recall(B)"]]
df = df.rename(
    columns={
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_classification_loss",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
    }
)

df.plot(
    subplots=True,
    layout=(2, 2),
    title="Object Detection training results",
    figsize=(10, 4),
)
Path(runs_dir + "/detect/train2/plots/").mkdir(parents=True, exist_ok=True)
plt.savefig(runs_dir + "/detect/train2/plots/results.jpg")
