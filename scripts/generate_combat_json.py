#!/usr/bin/env python3
"""
generate_combat_json.py

Automatically generates a Label Studio JSON file for all MP4 videos
in a specified folder. Each video gets an entry with empty annotations.
"""

import os
import json

# --- CONFIGURE THESE PATHS FOR YOUR MACHINE ---
video_folder = r"C:\Users\v5karthi\Desktop\converted_mp4"
output_json_path = r"C:\Users\v5karthi\Desktop\judo-footage-analysis-main\data\combat_phase\project.json"


# --- SCRIPT ---
def generate_json(video_folder, output_json_path):
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    if not videos:
        print(f"No MP4 files found in {video_folder}")
        return

    data = [{"video": os.path.join(video_folder, v), "annotations": []} for v in videos]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON file created with {len(videos)} videos at: {output_json_path}")


if __name__ == "__main__":
    generate_json(video_folder, output_json_path)
