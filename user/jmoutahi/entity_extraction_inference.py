import json
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


class OverlayBoxesOnVideo(luigi.Task):
    input_video_path = luigi.Parameter()
    input_json_path = luigi.Parameter()
    output_video_path = luigi.Parameter()

    def run(self):
        with open(self.input_json_path, "r") as json_file:
            data = json.load(json_file)

        cap = cv2.VideoCapture(self.input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Write as a mp4 file
        out = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        box_color = {
            "player_blue": (255, 0, 0),
            "player_white": (0, 0, 255),
            "referee": (0, 255, 0),
        }

        for i in range(num_frames):
            ret, frame = cap.read()
            for entity in data:
                if entity["frame"] == i:
                    x1, y1, x2, y2 = (
                        entity["x1"],
                        entity["y1"],
                        entity["x2"],
                        entity["y2"],
                    )
                    aspect_ratio = entity["aspect_ratio"]
                    confidence = entity["confidence"]
                    start_pt = (x1, y1)
                    end_pt = (x2, y2)
                    color = box_color[entity["name"]]
                    thickness = 2
                    cv2.rectangle(frame, start_pt, end_pt, color, thickness)
                    cv2.putText(
                        frame,
                        entity["name"],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"AR: {aspect_ratio:.2f}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
                    # display confidence
                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.2f}",
                        (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
            out.write(frame)

        cap.release()
        out.release()

    def output(self):
        return luigi.LocalTarget(self.output_video_path)


class FullWorkflow(luigi.Task):
    input_video_path = luigi.Parameter()
    json_output_path = luigi.Parameter()
    video_output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def requires(self):
        return SceneEntityExtractionInference(
            input_video_path=self.input_video_path,
            json_output_path=self.json_output_path,
            checkpoint=self.checkpoint,
        )

    def run(self):
        yield OverlayBoxesOnVideo(
            input_video_path=self.input_video_path,
            input_json_path=self.json_output_path,
            output_video_path=self.video_output_path,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-video-path",
        type=str,
        # default="/home/GTL/jmoutahi/Desktop/IRML/test_video/test.mp4",
        default="/mnt/students/video_judo/interim/mat-2-trunc.mp4",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--json-output-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/entity_extraction_inference/output.json",
        help="Path to save the JSON output.",
    )
    parser.add_argument(
        "--video-output-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/entity_extraction_inference/output_video.mp4",
        help="Path to save the annotated video output.",
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
                input_video_path=args.input_video_path,
                json_output_path=args.json_output_path,
                video_output_path=args.video_output_path,
                checkpoint=args.checkpoint,
            )
        ],
        workers=args.num_workers,
    )
