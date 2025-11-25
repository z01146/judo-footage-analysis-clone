from argparse import ArgumentParser
from pathlib import Path

import cv2
import luigi
from ultralytics import YOLO

from judo_footage_analysis.utils import ensure_path


class RefereeExtraction(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def process_batch(self, paths, model):
        imgs = [cv2.imread(p.as_posix()) for p in paths]

        results = model.predict(
            imgs,
            save=False,
            conf=0.2,
            iou=0.5,
            verbose=True,
            stream=False,
            device="cpu",
        )

        for path, img, r in zip(paths, imgs, results):
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = box.cls[0]

                if not int(cls) == 2:
                    continue
                referee = img[y1 : y1 + h, x1 : x1 + w]
                filename = ensure_path(self.output_path) / f"{path.stem}_{i:02d}.png"
                cv2.imwrite(filename.as_posix(), referee)

    def glob_path_batches(self, input_path, pattern="*.jpg", batch_size=10):
        batch = []
        for p in sorted(Path(input_path).glob(pattern)):
            batch.append(p)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def run(self):
        model = YOLO(self.checkpoint)

        for i, batch in enumerate(self.glob_path_batches(self.input_path)):
            self.process_batch(batch, model)

        with self.output().open("w") as f:
            f.write("")


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
        default="/cs-share/pradalier/tmp/judo/data/referee_v2/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--scheduler-host", type=str, default="localhost")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))

    luigi.build(
        [
            RefereeExtraction(
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
        scheduler_host=args.scheduler_host,
    )
