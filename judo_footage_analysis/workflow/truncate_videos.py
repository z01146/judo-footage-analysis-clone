"""
Script for truncating long judo videos into smaller segments.

This version ensures ffmpeg/ffprobe is correctly found using imageio_ffmpeg.
"""

from argparse import ArgumentParser
from pathlib import Path

import ffmpeg
import luigi
import imageio_ffmpeg as iio_ffmpeg

from judo_footage_analysis.utils import ensure_path


class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    duration = luigi.IntParameter(default=600)  # duration of each segment in seconds

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        # Get the ffmpeg executable path
        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()

        # Probe video to get its total duration
        probe = ffmpeg.probe(self.input_path, cmd=ffmpeg_path)
        total_duration = int(float(probe["format"]["duration"]))

        # Calculate how many segments we need
        num_segments = (total_duration + self.duration - 1) // self.duration  # ceiling division

        for i in range(num_segments):
            start_time = i * self.duration
            output_file = ensure_path(self.output_path) / f"{i:04d}.mp4"

            (
                ffmpeg.input(self.input_path, ss=start_time, t=self.duration)
                .output(output_file.as_posix(), format="mp4")
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True, cmd=ffmpeg_path)
            )

        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="match")
    parser.add_argument("--duration", type=int, default=600, help="Duration of each segment in seconds")
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))

    tasks = [
        TruncateVideos(
            input_path=str(video),
            output_path=str(Path(args.output_root_path) / f"{args.output_prefix}_{i+1:02d}"),
            duration=args.duration,
        )
        for i, video in enumerate(videos)
    ]

    luigi.build(tasks, workers=args.num_workers)
