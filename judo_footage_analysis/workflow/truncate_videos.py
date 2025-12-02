"""
Workflow for sampling frames from livestream judo videos.

This script truncates videos into smaller segments for analysis.
It uses FFmpeg via a local binary if needed.
"""

from argparse import ArgumentParser
from pathlib import Path
import os

import ffmpeg
import luigi
from judo_footage_analysis.utils import ensure_path

# --- PATCHED: Explicit FFmpeg / FFprobe paths ---
FFMPEG_BIN_DIR = r"C:\Users\v5karthi\Desktop\ffmpeg-8.0.1-essentials_build\bin"
FFMPEG_PATH = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

# Add bin folder to PATH so ffmpeg can be found by subprocess
os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
# --------------------------------------------------


class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    # FFmpeg parameters
    offset = luigi.IntParameter(default=0)  # seconds to skip at start
    duration = luigi.IntParameter(default=30)  # duration of each truncated segment
    num_truncations = luigi.IntParameter(default=5)  # number of segments to create

    def output(self):
        """Check for a success semaphore."""
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        """Truncate input video into segments."""
        # Ensure output path exists
        out_dir = ensure_path(self.output_path)

        # Get video duration
        probe = ffmpeg.probe(self.input_path, cmd=FFPROBE_PATH)
        total_duration = int(float(probe["format"]["duration"]))

        # Calculate how many truncations to do if not specified
        truncations = self.num_truncations or max(1, total_duration // self.duration)

        for i in range(truncations):
            start_time = self.offset + i * self.duration
            if start_time >= total_duration:
                break

            output_file = out_dir / f"{i:04d}.mp4"
            (
                ffmpeg.input(self.input_path, ss=start_time, t=self.duration, cmd=FFMPEG_PATH)
                .output(output_file.as_posix(), format="mp4")
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True, cmd=FFMPEG_PATH)
            )

        # Write _SUCCESS file
        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="match")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))

    luigi.build(
        [
            TruncateVideos(
                input_path=str(v),
                output_path=str(Path(args.output_root_path) / f"{args.output_prefix}_{i+1:02d}"),
                duration=args.duration,
            )
            for i, v in enumerate(videos)
        ],
        workers=args.num_workers,
        local_scheduler=True,  # avoids requiring a central Luigi scheduler
    )
