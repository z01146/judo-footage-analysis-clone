"""Script for sampling frames from livestream judo videos.

In this particular script, we are generating frames that we will use for whole
scene classification. We will sample at 1hz, and place the resulting frames into
a directory structure that should be relatively easy to retrieve for our
labeling tasks.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import luigi

from judo_footage_analysis.utils import ensure_path


class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    # ffmpeg parameters
    offset = luigi.IntParameter(default=60 * 60)
    duration = luigi.IntParameter(default=30)
    num_truncations = luigi.IntParameter(default=5)

    def output(self):
        """Check for a success semaphore."""
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        """Dump frames from the input video to the output directory, prefixed by modulo of the frame number."""
        # get the length of the video
        probe = ffmpeg.probe(self.input_path)
        duration = int(float(probe["format"]["duration"]))

        for i in range(self.num_truncations):
            output_file = ensure_path(self.output_path) / f"{i:04d}.mp4"
            (
                ffmpeg.input(
                    self.input_path, ss=self.offset + i * self.duration, t=self.duration
                )
                .output(output_file.as_posix(), format="mp4")
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )

        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="mat")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))
    luigi.build(
        [
            TruncateVideos(
                input_path=p.as_posix(),
                output_path=(
                    Path(args.output_root_path) / f"{args.output_prefix}_{i+1:02d}"
                ).as_posix(),
                duration=args.duration,
            )
            for i, p in enumerate(videos)
        ],
        workers=args.num_workers,
    )
