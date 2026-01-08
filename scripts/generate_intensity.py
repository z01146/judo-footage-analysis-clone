import pandas as pd
import numpy as np
import argparse
import ast


def extract_center(detection_str):
    try:
        # Parse the string representation of detections
        detections = ast.literal_eval(detection_str)
        if not detections or len(detections) == 0:
            return np.nan, np.nan

        # Extract x1, y1, x2, y2 from the first detection
        d = detections[0]
        cx = (d[0] + d[2]) / 2
        cy = (d[1] + d[3]) / 2
        return cx, cy
    except:
        return np.nan, np.nan


def generate_intensity(input_file, output_file):
    df = pd.read_csv(input_file)
    print(f"Processing columns: {df.columns.tolist()}")

    # 1. Extract coordinates (Returning NaN instead of None for math stability)
    coords = df['detections'].apply(lambda x: pd.Series(extract_center(x)))
    df['center_x'], df['center_y'] = coords[0], coords[1]

    # 2. Fill gaps: If a frame is empty, assume the player stayed in the last known spot
    df['center_x'] = df['center_x'].ffill().fillna(0)
    df['center_y'] = df['center_y'].ffill().fillna(0)

    # 3. Calculate Velocity (Distance between frames)
    df['dx'] = df['center_x'].diff().fillna(0)
    df['dy'] = df['center_y'].diff().fillna(0)

    # 4. Intensity Score = sqrt(dx^2 + dy^2)
    df['intensity_score'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

    # 5. Smoothing (30 frame window for cleaner trends)
    df['smoothed_intensity'] = df['intensity_score'].rolling(window=30, min_periods=1).mean()

    df.to_csv(output_file, index=False)
    print(f"Intensity mapping complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    generate_intensity(args.input_file, args.output_file)