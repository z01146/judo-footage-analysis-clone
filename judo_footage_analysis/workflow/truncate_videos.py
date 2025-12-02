"""
Script for segmenting long Judo videos into shorter clips for analysis.

This script splits videos into fixed-duration segments using ffmpeg.
FFmpeg is loaded from imageio_ffmpeg to avoid missing binary errors on Windows.
"""

from argparse import ArgumentParser
from pathlib import Path

import ffmpeg
import luigi
from imageio_ffmpeg import get_ffmpeg_exe

from judo_footage_analysis.utils import ensure_path


class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    offset = luigi.IntParameter(default=0)  # start at 0 seconds
    duration = luigi.IntParameter(default=600)  # segment length in seconds
    num_truncations = luigi.IntParameter(default=None)  # calculated if None

    def output(self):
        """Check for a success semaphore."""
        return luigi.LocalTarget(str(Path(self.output_path) / "_SUCCESS"))

    def run(self):
        ffmpeg_path = get_ffmpeg_exe()  # ensures correct FFmpeg binary is used

        # Probe video to get total duration
        probe = ffmpeg.probe(self.input_path, cmd=ffmpeg_path)
        total_duration = float(probe["format"]["duration"])

        # Calculate number of truncations if not explicitly set
        num_truncations = (
            self.num_truncations
            if self.num_truncations is not None
            else int((total_duration - self.offset) // self.duration) + 1
        )

        output_dir = ensure_path(self.output_path)

        for i in range(num_truncations):
            start_time = self.offset + i * self.duration
            output_file = output_dir / f"{i+1:04d}.mp4"

            ffmpeg.input(self.input_path, ss=start_time, t=self.duration).output(
                output_file.as_posix(), format="mp4"
            ).run(overwrite_output=True, cmd=ffmpeg_path)

        # Create success file
        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="match")
    parser.add_argument("--duration", type=int, default=600, help="Segment duration in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))

    tasks = [
        TruncateVideos(
            input_path=video.as_posix(),
            output_path=(Path(args.output_root_path) / f"{args.output_prefix}_{i+1:02d}").as_posix(),
            duration=args.duration,
        )
        for i, video in enumerate(videos)
    ]

    luigi.build(tasks, workers=args.num_workers)
