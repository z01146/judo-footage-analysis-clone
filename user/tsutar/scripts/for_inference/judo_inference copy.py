from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO


class Inference(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    # output_prefix = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path + "/runs")

    def run(self):
        # self.output_path.mkdir(parents=True, exist_ok=True)
        model_name = "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scriipts/runs/detect/train2/weights/best.pt"
        model = YOLO(model_name)

        print(self.input_path)
        predict = model(self.input_path, save=False, conf=0.2, iou=0.5, stream=True)

        # for d in self.input_path:
        #     print(d)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-root-path", type=str, default="/cs-share/pradalier/tmp/judo/frames/"
    )
    parser.add_argument(
        "--output-root-path", type=str, default="/cs-share/pradalier/tmp/judo/"
    )
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))
    # print("\n".join([p.parents[0].name + "/" + p.name for p in image_batch_root]))
    luigi.build(
        [
            Inference(
                input_path=p.as_posix(),
                output_path=args.output_root_path + p.parents[0].name + "/" + p.name,
            )
            for i, p in enumerate(image_batch_root)
        ],
        workers=args.num_workers,
    )
