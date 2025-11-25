import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_on_video(video_path, json_path, output_path="output.mp4", N=None):
    """Create a video with the data from the json on the video"""
    # Load the json in a pandas dataframe
    df = pd.read_json(json_path)
    print(df.head())
    print(df.columns)

    # Print the filename column
    print(df["filename"].values[0])

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"frame_width: {frame_width}, frame_height: {frame_height}, fps: {fps}, number_of_frames: {number_of_frames}, number_of_frames in df: {len(df)}"
    )

    # Verify that the video has as many frames as the json
    assert len(df) == number_of_frames

    # Create a VideoWriter object
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    # Iterate over the dataframe
    i = 0
    for index, row in df.iterrows():
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the text on the frame
        text = [
            f"Frame {index}",
            f"Filename : {row['filename']}",
            f"Available : {row['available']}",
            f"Raw Text : {row['raw_text']}",
            f"Not processed timer : {row['time_seconds']}",
            f"Processed timer : {row['time_seconds_filled']}",
            "Not processed : " + "Paused"
            if row["time_seconds_derivative"] == 0
            else "Running",
            "Processed : " + "Paused"
            if row["time_seconds_derivative_over150frames"] == 0
            else "Running",
        ]

        for j, t in enumerate(text):
            cv2.putText(
                frame,
                t,
                (10, 30 * (j + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Write the frame
        out.write(frame)

        i += 1
        # Print the percentage of the video processed
        if i * 100 / number_of_frames % 10 == 0:
            print(f"{i * 100 / number_of_frames}%")

        if N is not None and i > N:
            break

    # Release the VideoCapture and the VideoWriter
    cap.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()


# folder = "json_videos"
# json_filename = "timer_[Mat+6]+2023+President's+Cup+[B38xef6cHHk].mp4_filled.json"
# json_path = os.path.join(folder, json_filename)

# data_on_video("/mnt/students/video_judo/[Mat+6]+2023+President's+Cup+[B38xef6cHHk].mp4", json_path)


def data_as_subtitles(video_path, json_path, output_path="output.srt", N=None):
    """Create a subtitle file (SRT) with the data from the json on the video (One subtitle per second)"""

    # Create the SRT file
    with open(output_path, "w") as f:
        # Load the json in a pandas dataframe
        df = pd.read_json(json_path)

        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Verify that the video has as many frames as the json
        assert len(df) == number_of_frames

        # Iterate over the dataframe
        i = 0
        for index, row in df.iterrows():
            if index % fps != 0:
                continue

            # Write the subtitle
            f.write(f"{i+1}\n")
            time = index // fps
            time = pd.to_datetime(time, unit="s").strftime(
                "%H:%M:%S,000"
            )  # 00:00:00,000
            # Write the time
            f.write(f"{time} --> ")
            time = (index) // fps + 1
            time = pd.to_datetime(time, unit="s").strftime("%H:%M:%S,000")
            f.write(f"{time}\n")

            # Write the text
            text = [
                f"Frame {index}",
                f"Filename : {row['filename']}",
                f"Available : {row['available']}",
                f"Raw Text : {row['raw_text']}",
                f"Not processed timer : {row['time_seconds']}",
                f"Processed timer : {row['time_seconds_filled']}",
                "Not processed : Paused"
                if row["time_seconds_derivative"] == 0
                else "Not processed : Running",
                "Processed : Paused"
                if row["time_seconds_derivative_over150frames"] == 0
                else "Processed : Running",
            ]

            for t in text:
                f.write(t + "\n")

            f.write("\n")

            i += 1

            if N is not None and i > N:
                break

        # Release the VideoCapture
        cap.release()
