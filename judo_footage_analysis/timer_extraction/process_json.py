import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_json(
    timer_json="combined_videos.json",
    filename="timer_filled.json",
    frame_rate=30,
    verbose=False,
):
    # Load the json in a pandas dataframe
    df = pd.read_json(timer_json)

    # if(verbose) : the first 5 rows of the dataframe
    if verbose:
        print(df.head())

    # Print the columns names
    if verbose:
        print(df.columns)

    # Sort by the filename
    df.sort_values(by="filename", inplace=True)

    df["mat_number"] = df["filename"].apply(lambda x: x[25:32])
    df["number"] = df["filename"].apply(lambda x: x[30:31])
    if verbose:
        print(df.head(1))

    # For each mat, sort the frames by the frame number in ascending order
    df = (
        df.groupby("mat_number")
        .apply(lambda x: x.sort_values("frame_number", ascending=True))
        .reset_index(drop=True)
    )

    # Create a new column to store the time in seconds
    df["time_seconds"] = df["minutes"] * 60 + df["seconds"]
    if verbose:
        print(df.head())

    # Fill the missing values smartly
    L = df["time_seconds"].copy().values

    # Find the holes
    i = 0
    while i < len(L):
        # print(f"i = {i}")
        if np.isnan(L[i]):
            beg = i
            while i < len(L) and np.isnan(L[i]):
                i += 1
            end = i

            # print(f"Found a hole from {beg} to {end}")
            if end == len(L):
                time_diff = L[end - 1] - L[beg - 1]
            else:
                time_diff = L[end] - L[beg - 1]
            hole_size = end - beg
            expected_time_diff = hole_size / frame_rate
            while expected_time_diff > time_diff and hole_size > 0:
                # Fill with the previous value
                L[beg] = L[beg - 1]
                beg += 1
                hole_size -= 1
            # Fill the rest with interpolation
            if end == len(L):
                L[beg:end] = np.linspace(L[beg - 1], L[end - 1], end - beg)
            else:
                L[beg:end] = np.linspace(L[beg - 1], L[end], end - beg)
            i = end
        else:
            i += 1

    df["time_seconds_filled"] = L
    # Apply a rolling mean to smooth the time in seconds
    window = frame_rate
    df["time_seconds_filled"] = (
        df["time_seconds_filled"].rolling(window, center=True).mean()
    )
    if verbose:
        print(df.head())

    # Compute the derivative of the time
    df["time_seconds_derivative"] = df["time_seconds_filled"].diff()
    # if the derivative is negative, set it to -1 otherwise 0
    df["time_seconds_derivative"] = df["time_seconds_derivative"].apply(
        lambda x: -1 if x < 0 else 0
    )
    if verbose:
        print(df.head())

    # If the derivative is 0 for less than N frames, set it to -1
    N = frame_rate * 3
    L = df["time_seconds_derivative"].copy().values
    i = 0
    while i < len(L):
        if L[i] == 0:
            beg = i
            while i < len(L) and L[i] == 0:
                i += 1
            end = i
            # print(f"Found a zero derivative from {beg} to {end}")
            if end - beg < N:
                L[beg:end] = -1
            i = end
        else:
            i += 1
    df["time_seconds_derivative_over" + str(N) + "frames"] = L
    if verbose:
        print(df.head())

    N = frame_rate * 5
    L = df["time_seconds_derivative"].copy().values
    i = 0
    while i < len(L):
        if L[i] == 0:
            beg = i
            while i < len(L) and L[i] == 0:
                i += 1
            end = i
            # print(f"Found a zero derivative from {beg} to {end}")
            if end - beg < N:
                L[beg:end] = -1
            i = end
        else:
            i += 1
    df["time_seconds_derivative_over" + str(N) + "frames"] = L
    if verbose:
        print(df.head())

    N = frame_rate * 10
    L = df["time_seconds_derivative"].copy().values
    i = 0
    while i < len(L):
        if L[i] == 0:
            beg = i
            while i < len(L) and L[i] == 0:
                i += 1
            end = i
            # print(f"Found a zero derivative from {beg} to {end}")
            if end - beg < N:
                L[beg:end] = -1
            i = end
        else:
            i += 1
    df["time_seconds_derivative_over" + str(N) + "frames"] = L
    if verbose:
        print(df.head())

    N = frame_rate * 15
    L = df["time_seconds_derivative"].copy().values
    i = 0
    while i < len(L):
        if L[i] == 0:
            beg = i
            while i < len(L) and L[i] == 0:
                i += 1
            end = i
            # print(f"Found a zero derivative from {beg} to {end}")
            if end - beg < N:
                L[beg:end] = -1
            i = end
        else:
            i += 1
    df["time_seconds_derivative_over" + str(N) + "frames"] = L
    if verbose:
        print(df.head())

    # Save the dataframe to a new json file
    df.to_json(filename)


def plot_timer(timer_json="timer_filled.json", frame_rate=30):
    # Load the json in a pandas dataframe
    print(f"Loading {timer_json}...")
    df = pd.read_json(timer_json)
    print(df.head(1))

    list_N = [frame_rate * 3, frame_rate * 5, frame_rate * 10, frame_rate * 15]

    # Plot the time in seconds and the derivative one two different subplots
    print("Plotting the time in seconds and its derivative...")
    for mat_number, mat_df in df.groupby("mat_number"):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(
            mat_df["time_seconds"].values,
            "g*",
            label="Original timer extracted using Tesseract",
        )
        ax[0].plot(mat_df["time_seconds_filled"].values, "r-", label="Filled timer")
        ax[0].set_title(f"Time in seconds for mat {mat_number}")
        ax[0].legend()
        ax[0].xaxis.set_label_text("Time in seconds")
        ax[0].yaxis.set_label_text("Timer value")
        ax[1].plot(
            mat_df["time_seconds_derivative"].values,
            "b-",
            label="Derivative of the timer in seconds",
        )
        for N in list_N:
            ax[1].plot(
                mat_df["time_seconds_derivative_over" + str(N) + "frames"].values,
                label=f"Derivative of the timer in seconds over {N} frames",
            )
        ax[1].set_title(f"Derivative of the time in seconds for mat {mat_number}")
        ax[1].legend()
        ax[1].xaxis.set_label_text("Time in seconds")
        ax[1].yaxis.set_label_text("Derivative of the timer value")
        plt.tight_layout()
        plt.show()

    print("Done!")
