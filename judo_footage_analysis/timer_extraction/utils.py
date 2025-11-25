import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract

from .combine_jsons import *
from .display_on_video import *
from .is_timer import *
from .process_json import *
from .timer_task import *

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"/usr/bin/tesseract"  # TODO: Change this to the Tessaract path on the system
)


# --------------------- Pre processing ---------------------
def process_all_frames(
    folder="/cs-share/pradalier/tmp/judo/frames/",
    output_folder="json/",
    output_filename="combined.json",
):
    # Specify the folder path containing frames
    frames_root_folder = folder

    json_folder = output_folder

    # Specify the region of interest as (x1, y1, x2, y2)
    roi_coordinates_timer = (660, 625, 750, 665)

    # Get the list of folders containing frames
    frame_mat_folders = [
        os.path.join(frames_root_folder, folder)
        for folder in os.listdir(frames_root_folder)
        if os.path.isdir(os.path.join(frames_root_folder, folder))
    ]
    frame_folders = [
        os.path.join(frame_mat_folder, folder)
        for frame_mat_folder in frame_mat_folders
        for folder in os.listdir(frame_mat_folder)
        if os.path.isdir(os.path.join(frame_mat_folder, folder))
    ]

    # Extract the timer from the frames
    print("-------------------")
    print("Extracting timer...")
    print("-------------------")

    parallelize_extraction_timer(frame_folders, json_folder, roi_coordinates_timer)

    print("-------------------")
    print("Extraction done!")
    print("-------------------")

    # Combine the JSON files
    print("-------------------")
    print("Combining JSON files...")
    print("-------------------")

    folder_path = "json/"
    output_file = "combined.json"

    combine_json_files(json_folder, output_filename)

    print("-------------------")
    print("Combining done!")
    print("-------------------")


def process_all_videos(
    videos_root_folder="/mnt/students/video_judo/",
    json_folder="json_videos/",
    output_filename="combined_videos.json",
):
    # Specify the region of interest as (x1, y1, x2, y2)
    roi_coordinates_timer = (660, 625, 750, 665)

    # Get the list of video paths

    video_folders = [
        os.path.join(videos_root_folder, video)
        for video in os.listdir(videos_root_folder)
        if video.endswith(".mp4")
    ]
    print(video_folders)
    print(len(video_folders))

    # Extract the timer from the videos
    print("-------------------")
    print("Extracting timer...")
    print("-------------------")

    parallelize_extraction_timer_video(
        video_folders, json_folder, roi_coordinates_timer
    )

    print("-------------------")
    print("Extraction done!")
    print("-------------------")

    # Combine the JSON files
    print("-------------------")
    print("Combining JSON files...")
    print("-------------------")

    combine_json_files(json_folder, output_filename)

    print("-------------------")
    print("Combining done!")
    print("-------------------")


# --------------------- Post processing ---------------------
folder = "json_videos"


def process_all_json(folder):
    # For each json in the folder apply process_json
    for filename in os.listdir(folder):
        if filename.endswith(".mp4.json"):
            output_filename = filename.replace(".json", "_filled.json")
            print(f"Processing {filename}")
            process_json(
                os.path.join(folder, filename),
                os.path.join(folder, output_filename),
                verbose=False,
            )
            print("Done")
            print()


def plot_all_json(folder):
    # For each json in the folder plot the data
    for filename in os.listdir(folder):
        if filename.endswith("_filled.json"):
            print(f"Plotting {filename}")
            plot_timer(os.path.join(folder, filename))
            print("Done")
            print()


def create_all_videos(folder):
    # For each json in the folder create the video
    for filename in os.listdir(folder):
        if filename.endswith("_filled.json"):
            output_filename = filename.replace("_filled.json", "_output.mp4")
            print(f"Processing {filename}")
            # Load the json in a pandas dataframe
            df = pd.read_json(os.path.join(folder, filename))
            video_path = df["filename"].values[0]
            # remove df from the memory
            del df
            # Get the video name (after the last /)
            video_name = video_path.split("/")[-1]
            json_path = os.path.join(folder, filename)

            # Create the video
            data_on_video(
                video_path,
                json_path,
                output_path=os.path.join(folder, output_filename),
                N=1000,
            )
            print("Done")
            print()


def create_all_srt(folder):
    # For each json in the folder create the SRT file
    for filename in os.listdir(folder):
        if filename.endswith("_filled.json"):
            output_filename = filename.replace("_filled.json", "_output.srt")
            print(f"Processing {filename}")
            # Load the json in a pandas dataframe
            df = pd.read_json(os.path.join(folder, filename))
            video_path = df["filename"].values[0]
            # remove df from the memory
            del df
            json_path = os.path.join(folder, filename)

            # Create the SRT file
            data_as_subtitles(
                video_path, json_path, output_path=os.path.join(folder, output_filename)
            )
            print("Done")
            print()
