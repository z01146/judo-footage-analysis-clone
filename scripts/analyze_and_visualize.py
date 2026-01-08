import pandas as pd
import numpy as np
import argparse
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def extract_center(detection_str):
    """
    Parses detection strings and selects the largest bounding box (primary athletes)
    to calculate a central motion point.
    """
    try:
        if isinstance(detection_str, str):
            detections = ast.literal_eval(detection_str)
        else:
            detections = detection_str

        if not detections or len(detections) == 0:
            return np.nan, np.nan

        # Select the detection with the largest area to lock onto active athletes
        largest_box = max(
            detections,
            key=lambda x: (float(x[2]) - float(x[0])) * (float(x[3]) - float(x[1]))
        )

        cx = (float(largest_box[0]) + float(largest_box[2])) / 2
        cy = (float(largest_box[1]) + float(largest_box[3])) / 2
        return cx, cy
    except Exception:
        return np.nan, np.nan

def run_full_analysis(input_file, output_csv):
    df = pd.read_csv(input_file)

    # 1. Coordinate Extraction
    coords = df['detections'].apply(lambda x: pd.Series(extract_center(x)))
    df['center_x'], df['center_y'] = coords[0], coords[1]

    # Debug info
    valid_coords = df['center_x'].count()
    print(f"Total rows: {len(df)} | Rows with valid detections: {valid_coords}")

    # Fill gaps for continuity
    df['center_x'] = df['center_x'].ffill().bfill().fillna(0)
    df['center_y'] = df['center_y'].ffill().bfill().fillna(0)

    # 2. Intensity Calculation
    df['dx'] = df['center_x'].diff().fillna(0)
    df['dy'] = df['center_y'].diff().fillna(0)
    df['intensity_score'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

    # Boost normalized coordinates
    if 0 < df['intensity_score'].max() < 1.0:
        print("Detected normalized coordinates. Scaling intensity...")
        df['intensity_score'] *= 1000

    df['smoothed_intensity'] = df['intensity_score'].rolling(window=30, min_periods=1).mean()
    print(f"Max Intensity detected: {df['smoothed_intensity'].max()}")

    df.to_csv(output_csv, index=False)

    # 3. Visualization
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(15, 8))

    x_axis = np.arange(len(df))
    colors = {'Tachi-waza': '#2ecc71', 'Ne-waza': '#3498db', 'Mate': '#e74c3c', 'No-Match/Intermission': '#95a5a6'}

    # Background Phase Coloring
    for phase in df['phase'].unique():
        mask = df['phase'] == phase
        plt.fill_between(x_axis, df['smoothed_intensity'],
                         where=mask, color=colors.get(phase, '#bdc3c7'),
                         alpha=0.3, label=phase)

    # Plot Intensity
    plt.plot(x_axis, df['smoothed_intensity'], color='black', linewidth=1.2, label='Intensity', alpha=0.8)

    # --- ACTION LABELER ---
    # Find top 0.5% intensity peaks to label as major actions
    threshold = df['smoothed_intensity'].quantile(0.995)
    if threshold > 0:
        # Finding local peaks to avoid labeling every single frame of a high-action moment
        peaks = df[(df['smoothed_intensity'] > threshold)].iloc[::50]
        for idx, row in peaks.iterrows():
            plt.annotate('MAJOR ACTION',
                         xy=(idx, row['smoothed_intensity']),
                         xytext=(idx, row['smoothed_intensity'] + (df['smoothed_intensity'].max() * 0.1)),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         fontsize=9, fontweight='bold', ha='center')

    # Scaling and Labels
    max_val = df['smoothed_intensity'].max()
    plt.ylim(0, max_val * 1.3 if max_val > 0 else 1.0)
    plt.title('Judo Combat Phase: Intensity Mapping & Action Detection', fontsize=16)
    plt.xlabel('Match Timeline (Frames)', fontsize=12)
    plt.ylabel('Activity Level (Motion Velocity)', fontsize=12)

    # Clean up legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)

    output_path = output_csv.replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Process Complete. Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    run_full_analysis(args.input_file, args.output_file)