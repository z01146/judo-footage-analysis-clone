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


class FrameSampler(luigi.Task):
    input_path = luigi.Parameter()
    output_root_path = luigi.Parameter()
    output_prefix = luigi.OptionalStrParameter(default=None)

    # ffmpeg parameters
    offset = luigi.IntParameter(default=0)
    sample_rate = luigi.IntParameter(default=1)
    duration = luigi.OptionalIntParameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_path = Path(self.output_root_path)
        if self.output_prefix:
            self.output_path /= self.output_prefix

    def output(self):
        """We check for the success semaphore."""
        return luigi.LocalTarget(self.output_path / "_SUCCESS")

    def run(self):
        """Dump frames from the input video to the output directory, prefixed by modulo of the frame number."""
        source = ffmpeg.input(self.input_path, ss=self.offset)

        if self.duration is not None:
            source = source.trim(duration=self.duration)

        source.output(
            str(ensure_path(self.output_path) / "%04d.jpg"),
            r=self.sample_rate,
            start_number=0,
        ).run(capture_stdout=True, capture_stderr=True)

        # write a success semaphore
        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="mat")
    parser.add_argument("--offset", type=int, default=0, help="Offset in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample rate in Hz")
    parser.add_argument("--batch-size", type=int, default=60 * 10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def compute_batch_params(offset, duration, sample_rate, batch_size):
    """Compute the parameters for a batch of videos.

    We assume that the videos are all the same length, and that the offset and
    duration are the same for all videos.
    """
    n_frames = duration * sample_rate
    if batch_size > n_frames:
        raise ValueError(
            f"Batch size {batch_size} is larger than the number of frames {n_frames}"
        )
    n_batches = n_frames // batch_size

    for i in range(n_batches):
        # seq_id, offset, duration
        yield i, offset + i * batch_size, batch_size


if __name__ == "__main__":
    args = parse_args()
    videos = sorted(Path(args.input_root_path).glob("*.mp4"))
    luigi.build(
        [
            FrameSampler(
                input_path=p.as_posix(),
                output_root_path=args.output_root_path,
                output_prefix=f"{args.output_prefix}_{i+1:02d}/{seq_id:04d}",
                offset=offset,
                duration=duration,
                sample_rate=args.sample_rate,
            )
            for seq_id, offset, duration in compute_batch_params(
                args.offset, args.duration, args.sample_rate, args.batch_size
            )
            for i, p in enumerate(videos)
        ],
        workers=args.num_workers,
    )
