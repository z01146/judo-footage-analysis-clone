"""Script for sampling frames from livestream judo videos.

In this particular script, we are generating frames that we will use for whole
scene classification. We will sample at 1hz, and place the resulting frames into
a directory structure that should be relatively easy to retrieve for our
labeling tasks.
"""

import io
from argparse import ArgumentParser
from pathlib import Path

import ffmpeg
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from judo_footage_analysis.utils import ensure_parent


def get_ffmpeg_output_async(path: str, width: int, height: int, framerate: int):
    """Initialize the ffmpeg stream."""
    return (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=framerate,
        )
        # https://stackoverflow.com/questions/10241433/slow-ffmpegs-images-per-second-when-creating-video-from-images
        # this isn't the right way to do things, but i want something to work...
        # .filter("setpts", f"(25/{framerate})*PTS")
        .output(path, pix_fmt="yuv420p", vcodec="libx264")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


class ReconstructVideoClassificationInference(luigi.Task):
    input_frames_path = luigi.Parameter()
    input_inference_path = luigi.Parameter()
    prefix = luigi.Parameter()
    output_path = luigi.Parameter()
    framerate = luigi.IntParameter(default=2)

    @classmethod
    def _preprocess(self, df):
        # a bunch of pre-processing
        df["mat"] = df["path"].apply(lambda x: int(x.split("/")[-3].split("_")[-1]))
        df["batch_id"] = df["path"].apply(lambda x: int(x.split("/")[-2]))
        df["frame_id"] = df["path"].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
        df["timestamp"] = df.batch_id * 600 + df.frame_id
        df["predicted_index"] = df.prob.apply(lambda x: np.argmax(x))
        labels = df.iloc[0].labels
        df["predicted_label"] = df.predicted_index.apply(lambda x: labels[x])
        for label in labels:
            df[label] = df.prob.apply(lambda x: x[labels.index(label)])
        df = df.set_index("timestamp").sort_index()
        return df

    @classmethod
    def _pad_df(self, df, k=60):
        min_timestamp = df.index.min()
        max_timestamp = df.index.max()
        for i in range(min_timestamp - k, min_timestamp):
            # pad the dataframe with the first value
            df.loc[i] = df.iloc[0]
        for i in range(max_timestamp + 1, max_timestamp + k):
            # now pad the dataframe with the last value
            df.loc[i] = df.iloc[-1]
        df = df.sort_index()
        return df

    @classmethod
    def _plot(self, df, ax=None):
        if ax is None:
            ax = plt.gca()
        labels = df.iloc[0].labels
        mat_id = df.iloc[0].mat
        batch_id = df.iloc[0].batch_id
        for label in labels:
            df[label].plot(label=label, ax=ax)
        plt.gcf().set_facecolor("white")
        # legend should always be on the left
        plt.legend(loc="right")
        plt.title(f"Probabilities over time for mat {mat_id} batch {batch_id}")

    def output(self):
        return [
            # test image
            luigi.LocalTarget(
                (Path(self.output_path) / self.prefix / "sample.png").as_posix()
            ),
            # video
            luigi.LocalTarget(
                (Path(self.output_path) / self.prefix / "output.mp4").as_posix()
            ),
            # success file
            luigi.LocalTarget(
                (Path(self.output_path) / self.prefix / "_SUCCESS").as_posix()
            ),
        ]

    def run(self):
        df = pd.read_json(Path(self.input_inference_path) / self.prefix / "output.json")
        df = self._preprocess(df)
        min_timestamp = df.index.min()
        max_timestamp = df.index.max()
        k = 30
        df = self._pad_df(df, k)

        output_path = ensure_parent(self.output()[0].path).parent

        # now plot windows of 60 seconds, centered on the current timestamp
        outstream = None

        for i in range(min_timestamp, max_timestamp):
            window = df.loc[i - k : i + k]
            # let's create a subplot with the original image and the plot
            fig, ax = plt.subplots(
                3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [6, 1, 3]}
            )
            # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
            # read the image directly from the dataframe
            ax[0].imshow(plt.imread(df.loc[i].path))
            ax[0].axis("off")

            # let's create a table
            ax[1].axis("off")
            values = df.loc[i][-4:].tolist()
            table = ax[1].table(
                cellText=[[f"{v:.2f}" for v in values]],
                colLabels=df.columns[-4:],
                loc="center",
            )
            max_index = values.index(max(values))
            table[(0, max_index)].set_facecolor("lightgreen")
            table[(1, max_index)].set_facecolor("lightgreen")

            self._plot(window, ax[2])
            # put a red vertical line at the current timestamp
            plt.axvline(x=i, color="r", linestyle="--")
            plt.title(f"Probabilities over time (t={i:04d}s)")
            # plt.tight_layout()

            # only save the first frame
            if i == min_timestamp:
                plt.savefig(self.output()[0].path)

            # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
            # get the numpy buffer
            io_buf = io.BytesIO()
            plt.savefig(io_buf, format="raw")
            io_buf.seek(0)
            data = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            # drop the alpha channel
            data = data[:, :, :3]
            width = int(fig.bbox.bounds[2])
            height = int(fig.bbox.bounds[3])
            io_buf.close()
            plt.close()
            if outstream is None:
                outstream = get_ffmpeg_output_async(
                    self.output()[1].path, width, height, framerate=self.framerate
                )
            outstream.stdin.write(data.tobytes())
        outstream.stdin.close()
        outstream.wait()
        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-frames-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--input-inference-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/fullframe_inference/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/fullframe_overlay_v1/",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_frames_path).glob("*/*"))

    luigi.build(
        [
            ReconstructVideoClassificationInference(
                input_frames_path=args.input_frames_path,
                input_inference_path=args.input_inference_path,
                prefix=p.relative_to(Path(args.input_frames_path)).as_posix(),
                output_path=args.output_root_path,
            )
            for p in image_batch_root
        ],
        workers=args.num_workers,
    )
