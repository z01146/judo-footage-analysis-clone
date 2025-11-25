import json
from argparse import ArgumentParser
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from judo_footage_analysis.utils import ensure_parent


class SceneClassificationInference(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path + "/output.json")

    def run(self):
        model = YOLO(self.checkpoint)
        model_prediction = model.predict(
            self.input_path,
            save=False,
            conf=0.2,
            iou=0.5,
            verbose=False,
            stream=True,
        )

        results = []
        for pred in model_prediction:
            probs = pred.probs.cpu().numpy()
            results.append(
                dict(
                    labels=list(pred.names.values()),
                    path=pred.path,
                    prob=probs.data.tolist(),
                )
            )

        output_path = ensure_parent(self.output().path)
        with output_path.open("w") as outfile:
            json.dump(results, outfile)


class PlotClassificationInference(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            (Path(self.output_path) / "probability_plot.png").as_posix()
        )

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

    def _plot(self, df, path):
        labels = df.iloc[0].labels
        mat_id = df.iloc[0].mat
        batch_id = df.iloc[0].batch_id
        for label in labels:
            df[label].plot(label=label, figsize=(12, 4))
        plt.gcf().set_facecolor("white")
        plt.legend()
        plt.title(f"Probabilities over time for mat {mat_id} batch {batch_id}")
        plt.savefig(path)

    def run(self):
        df = pd.read_json(self.input_path)
        df = self._preprocess(df)
        self._plot(df, ensure_parent(self.output().path))


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def run(self):
        # inference plus plots
        task = SceneClassificationInference(
            input_path=self.input_path,
            output_path=self.output_path,
            checkpoint=self.checkpoint,
        )
        yield task
        yield PlotClassificationInference(
            input_path=task.output().path,
            output_path=self.output_path,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/fullframe_inference/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/cs-share/pradalier/tmp/judo/models/fullframe_classification/v2/train/weights/best.pt",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))

    luigi.build(
        [
            Workflow(
                input_path=p.as_posix(),
                output_path=(
                    Path(args.output_root_path)
                    / p.relative_to(Path(args.input_root_path))
                ).as_posix(),
                checkpoint=args.checkpoint,
            )
            for p in image_batch_root
        ],
        workers=args.num_workers,
    )
