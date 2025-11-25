import json
import os

import cv2
import ffmpeg
import numpy as np
import pandas as pd

path = "/cs-share/pradalier/tmp/judo/data/timer/json_videos/"
json_files = [f for f in os.listdir(path) if f.endswith("_filled.json")]

test = json_files[1]
print(test)

with open(path + test) as f:
    df = pd.read_json(f)

# print the unique values in the time_seconds_derivative_over150frames column

print(df["time_seconds_derivative_over150frames"].unique())

# print(df.head())

# df = pd.read_json(path + "test.json")

# Find the individual Matches using the timer data
frame_rate, max_pause_duration = 30, 60
matches = []  # list of tuples (start, end)
start = None
end = None
count_since_last_not_zero = 0
for index, row in df.iterrows():
    # print("row", row["time_seconds_derivative_over150frames"], "index", index, "start", start, "end", end, "count", count_since_last_not_zero)
    if abs(row["time_seconds_derivative_over150frames"]) > 0:
        # print("Found a non-zero value at index", index)
        if start is None:
            start = index
    else:
        count_since_last_not_zero += 1
        if (
            start is not None
            and count_since_last_not_zero > max_pause_duration * frame_rate
        ):
            end = index
            # print("Found a match from", start, "to", end)
            matches.append((start, end))
            start = None
            end = None
            count_since_last_not_zero = 0

print(matches)

# For each match, count the number of frames where the derivative is zero and the number of frames where the derivative is not zero
counts = []
total_count_zero_ratio = 0
total_count_not_zero_ratio = 0
for match in matches:
    start, end = match
    count_zero = 0
    count_not_zero = 0
    for i in range(start, end):
        if abs(df.iloc[i]["time_seconds_derivative_over150frames"]) > 0:
            count_not_zero += 1
        else:
            count_zero += 1
    total_count_zero_ratio += count_zero / (count_zero + count_not_zero)
    total_count_not_zero_ratio += count_not_zero / (count_zero + count_not_zero)

    counts.append(
        (
            count_zero,
            count_not_zero,
            count_zero / (count_zero + count_not_zero),
            count_not_zero / (count_zero + count_not_zero),
        )
    )

print("Average ratio of zero frames:", total_count_zero_ratio / len(matches))
print("Average ratio of non-zero frames:", total_count_not_zero_ratio / len(matches))

# Make violin plots of the counts
import matplotlib.pyplot as plt
import seaborn as sns

df_counts = pd.DataFrame(
    counts, columns=["count_zero", "count_not_zero", "zero_ratio", "not_zero_ratio"]
)
print(df_counts.head())
sns.violinplot(data=df_counts[["zero_ratio", "not_zero_ratio"]])
plt.show()
