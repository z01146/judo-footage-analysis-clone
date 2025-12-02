"""
Workflow for sampling frames from livestream judo videos.

This script truncates videos into smaller segments for analysis.
It uses FFmpeg via a local binary (FFmpeg 8.0.1) to avoid admin issues.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import math

import ffmpeg
import luigi
from judo_footage_analysis.utils import ensure_path

# --- Use local FFmpeg 8.0.1 binaries ---
FFMPEG_BIN_DIR = r"C:\Users\v5karthi\Desktop\ffmpeg-8.0.1-essentials_build\bin"
FFMPEG_PATH = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

# Add bin folder to PATH so ffmpeg-python can find it
os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
# --------------------------------------------------


class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_root = luigi.Parameter()  # Root folder for all segments
    output_prefix = luigi.Parameter(default="match")

    offset = luigi.IntParameter(default=0)       # seconds to skip at start
    duration = luigi.IntParameter(default=600)   # default 10 minutes per clip
    clips_per_folder = luigi.IntParameter(default=6)  # number of clips per folder

    @property
    def output_path(self):
        """Folder where video segments will be saved."""
        base_name = Path(self.input_path).stem
        return Path(self.output_root) / f"{self.output_prefix}_{base_name}"

    def output(self):
        """Dummy output for Luigi task completion."""
        return luigi.LocalTarget(self.output_path / "_SUCCESS")

    def run(self):
        """Truncate input video into segments."""
        # Ensure output path exists
        out_dir = ensure_path(self.output_path)

        # Get video duration
        try:
            probe = ffmpeg.probe(self.input_path, cmd=FFPROBE_PATH)
            total_duration = int(float(probe["format"]["duration"]))
        except ffmpeg.Error as e:
            print("FFprobe error:")
            print(e.stderr.decode())
            raise

        # Calculate how many truncations to do
        truncations = max(1, math.ceil(total_duration / self.duration))

        for i in range(truncations):
            start_time = self.offset + i * self.duration
            if start_time >= total_duration:
                break

            output_file = out_dir / f"{i:04d}.mp4"
            try:
                (
                    ffmpeg.input(self.input_path, ss=start_time, t=self.duration)
                    .output(str(output_file), vcodec='libx264', acodec='aac', format='mp4')
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True, cmd=FFMPEG_PATH)
                )
            except ffmpeg.Error as e:
                print(f"FFmpeg failed for segment {i}:")
                print(e.stderr.decode())
                raise

        # Write _SUCCESS file
        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="match")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds (default 10 mins)")
    parser.add_argument("--clips-per-folder", type=int, default=6, help="Number of clips per folder")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))

    tasks = []
    for v in videos:
        tasks.append(
            TruncateVideos(
                input_path=str(v),
                output_root=str(args.output_root_path),
                output_prefix=args.output_prefix,
                duration=args.duration,
                clips_per_folder=args.clips_per_folder,
            )
        )

    luigi.build(
        tasks,
        workers=args.num_workers,
        local_scheduler=True,
    )
