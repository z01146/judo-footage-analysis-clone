import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import ffmpeg
import luigi
import numpy as np
import tqdm
from ultralytics import YOLO


class SceneEntityExtractionInference(luigi.Task):
    input_video_path = luigi.Parameter()
    json_output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()
    classNames = ["player_blue", "player_white", "referee"]

    def output(self):
        return luigi.LocalTarget(self.json_output_path)

    def fetch_frames_batch(self, filename, batch_size=10):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        duration = float(video_info["duration"])
        framerate = int(eval(video_info["avg_frame_rate"]))
        width = int(video_info["width"])
        height = int(video_info["height"])

        input_stream = (
            ffmpeg.input(filename, ss=0, t=duration)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .run_async(pipe_stdout=True)
        )

        batch = []
        while True:
            in_bytes = input_stream.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            batch.append(in_frame)
            if batch_size == len(batch):
                yield batch
                batch = []
        if batch:
            yield batch

    def process_frames_batch(self, model, frames):
        return model.predict(
            frames, save=False, conf=0.2, iou=0.45, verbose=False, device="cpu"
        )

    def run(self):
        model = YOLO(self.checkpoint)
        model_prediction = []
        for batch in tqdm.tqdm(self.fetch_frames_batch(self.input_video_path)):
            model_prediction.extend(self.process_frames_batch(model, batch))

        results = []
        for pred, frame in zip(model_prediction, range(len(model_prediction))):
            for i, prediction in enumerate(pred.boxes):
                xyxy = prediction.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                w, h = x2 - x1, y2 - y1
                cls = prediction.cls[0]

                aspect_ratio = w / h
                name = self.classNames[int(cls)]

                results.append(
                    dict(
                        frame=frame,
                        name=name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        aspect_ratio=aspect_ratio,
                        confidence=float(prediction.conf[0]),
                    )
                )

        with self.output().open("w") as outfile:
            json.dump(results, outfile)

        with self.output().open("w") as outfile:
            json.dump(results, outfile)


class FullWorkflow(luigi.Task):
    input_video_folder = luigi.Parameter()
    json_output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def requires(self):
        video_folders = [
            f
            for f in os.listdir(self.input_video_folder)
            if os.path.isdir(os.path.join(self.input_video_folder, f))
        ]
        video_files = []
        for folder in video_folders:
            video_files.extend(
                [
                    os.path.join(folder, f)
                    for f in os.listdir(os.path.join(self.input_video_folder, folder))
                    if f.endswith(".mp4")
                ]
            )
        print("video_files: ", video_files)
        for video_file in video_files:
            yield SceneEntityExtractionInference(
                input_video_path=os.path.join(self.input_video_folder, video_file),
                json_output_path=os.path.join(
                    self.json_output_path, f"{os.path.splitext(video_file)[0]}.json"
                ),
                checkpoint=self.checkpoint,
            )

    def run(self):
        pass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-video-folder",
        type=str,
        # default="/home/GTL/jmoutahi/Desktop/IRML/test_video",
        default="/cs-share/pradalier/tmp/judo/data/clips",
        help="Path to the input video files.",
    )
    parser.add_argument(
        "--json-output-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/entity_extraction_inference/videos",
        help="Path to save the JSONs output.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt",
        help="Path to the YOLO model checkpoint.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of workers for Luigi."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [
            FullWorkflow(
                input_video_folder=args.input_video_folder,
                json_output_path=args.json_output_path,
                checkpoint=args.checkpoint,
            )
        ],
        workers=args.num_workers,
    )
