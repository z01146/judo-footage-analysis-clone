import json
import os

import cv2
import ffmpeg
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path = "/cs-share/pradalier/tmp/judo/data/combat_phase/project-16-at-2024-03-28-14-17-d0fa284f.json"
df = pd.read_json(path)
print(df.head())
print(df.columns)

annotations = df["annotations"]
datastu = df["data"]

extracted_annotations = []

for annotation, data in zip(annotations, datastu):
    extracted_annotation = {}
    print(annotation)
    print(type(annotation))
    annotation = annotation[0]
    print(annotation.keys())
    print(annotation["result"])
    results = annotation["result"]
    url = data["video_url"]
    print(url)
    file = os.path.join(
        "/cs-share/pradalier/tmp/judo/data/", "/".join(url.split("/")[-3:])
    )
    print(file)
    extracted_annotation["file"] = file
    extracted_annotation["annotations"] = []
    for result in results:
        print(result)
        print(result.keys())
        print(result["value"])
        value = result["value"]
        value["labels"] = value["labels"][0]
        extracted_annotation["annotations"].append(value)
    extracted_annotations.append(extracted_annotation)

print(extracted_annotations)

# Save the extracted annotations
with open(
    "/cs-share/pradalier/tmp/judo/data/combat_phase/extracted_annotations.json", "w"
) as outfile:
    json.dump(extracted_annotations, outfile)

# Load the extracted annotations
with open(
    "/cs-share/pradalier/tmp/judo/data/combat_phase/extracted_annotations.json", "r"
) as infile:
    extracted_annotations = json.load(infile)

# Filter the annotations
filtered_annotations = []
for extracted_annotation in extracted_annotations:
    file = extracted_annotation["file"]
    annotations = extracted_annotation["annotations"]
    filtered_annotations.append({"file": file, "annotations": []})
    match_intervals = []
    active_intervals = []
    standing_intervals = []
    for annotation in annotations:
        if annotation["labels"] == "Match":
            match_intervals.append((annotation["start"], annotation["end"]))
        elif annotation["labels"] == "Active":
            active_intervals.append((annotation["start"], annotation["end"]))
        elif annotation["labels"] == "Standing":
            standing_intervals.append((annotation["start"], annotation["end"]))

    # Verify that the rules for the intervals are respected and fix them if not
    # 1. Active intervals are included in match intervals
    # 2. Standing intervals are included in active intervals
    for match_interval in match_intervals:
        for i, active_interval in enumerate(active_intervals):
            if (
                active_interval[0] <= match_interval[0]
                and active_interval[1] <= match_interval[1]
                and active_interval[1] >= match_interval[0]
            ):
                active_intervals[i] = (match_interval[0], active_interval[1])
            elif (
                active_interval[0] >= match_interval[0]
                and active_interval[1] >= match_interval[1]
                and active_interval[0] <= match_interval[1]
            ):
                active_intervals[i] = (active_interval[0], match_interval[1])

    for i, active_interval in enumerate(active_intervals):
        for j, standing_interval in enumerate(standing_intervals):
            if (
                standing_interval[0] <= active_interval[0]
                and standing_interval[1] <= active_interval[1]
                and standing_interval[1] >= active_interval[0]
            ):
                standing_intervals[j] = (active_interval[0], standing_interval[1])
            elif (
                standing_interval[0] >= active_interval[0]
                and standing_interval[1] >= active_interval[1]
                and standing_interval[0] <= active_interval[1]
            ):
                standing_intervals[j] = (standing_interval[0], active_interval[1])

    # Add the labels "Not Active" and "Not Standing" intervals
    # 1. Not Active intervals are the intervals between the end of the last active interval and the start of the next active interval inside a match interval
    # 2. Not Standing intervals are the intervals between the end of the last standing interval and the start of the next standing interval inside an active interval
    def find_complement_intervals(interval, intervals):
        complement_intervals = []
        if len(intervals) == 0:
            complement_intervals.append(interval)
        else:
            if interval[0] < intervals[0][0]:
                complement_intervals.append((interval[0], intervals[0][0]))
            for i in range(len(intervals) - 1):
                complement_intervals.append((intervals[i][1], intervals[i + 1][0]))
            if interval[1] > intervals[-1][1]:
                complement_intervals.append((intervals[-1][1], interval[1]))
        return complement_intervals

    # Find not_active_intervals
    not_active_intervals = []
    for start, end in match_intervals:
        active_within_match = [
            (s, e) for s, e in active_intervals if start <= s <= e <= end
        ]
        complement_within_match = find_complement_intervals(
            (start, end), active_within_match
        )
        not_active_intervals.extend(complement_within_match)

    # Find not_standing_intervals
    not_standing_intervals = []
    for start, end in active_intervals:
        standing_within_active = [
            (s, e) for s, e in standing_intervals if start <= s <= e <= end
        ]
        complement_within_active = find_complement_intervals(
            (start, end), standing_within_active
        )
        not_standing_intervals.extend(complement_within_active)

    not_match_intervals = find_complement_intervals((0, 30), match_intervals)

    # Add the remaining intervals
    for match_interval in match_intervals:
        filtered_annotations[-1]["annotations"].append(
            {"labels": "Match", "start": match_interval[0], "end": match_interval[1]}
        )
    for active_interval in active_intervals:
        filtered_annotations[-1]["annotations"].append(
            {"labels": "Active", "start": active_interval[0], "end": active_interval[1]}
        )
    for standing_interval in standing_intervals:
        filtered_annotations[-1]["annotations"].append(
            {
                "labels": "Standing",
                "start": standing_interval[0],
                "end": standing_interval[1],
            }
        )
    for not_active_interval in not_active_intervals:
        filtered_annotations[-1]["annotations"].append(
            {
                "labels": "Not Active",
                "start": not_active_interval[0],
                "end": not_active_interval[1],
            }
        )
    for not_standing_interval in not_standing_intervals:
        filtered_annotations[-1]["annotations"].append(
            {
                "labels": "Not Standing",
                "start": not_standing_interval[0],
                "end": not_standing_interval[1],
            }
        )
    for not_match_interval in not_match_intervals:
        filtered_annotations[-1]["annotations"].append(
            {
                "labels": "Not Match",
                "start": not_match_interval[0],
                "end": not_match_interval[1],
            }
        )
print(filtered_annotations)

# Save the filtered annotations
with open(
    "/cs-share/pradalier/tmp/judo/data/combat_phase/filtered_annotations.json", "w"
) as outfile:
    json.dump(filtered_annotations, outfile)
