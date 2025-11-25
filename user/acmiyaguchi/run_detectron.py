#!/usr/bin/env python3
import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def parse_args():
    parser = ArgumentParser(description="run detectron2")
    parser.add_argument(
        "--config-file",
        type=str,
        default="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--model_device", type=str, default="cuda")
    parser.add_argument("--framerate", type=int, default=16)
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output-data-root", type=str)
    return parser.parse_args()


# def get_detectron_model(args):
#     """Initialize the detectron2 predictor."""
#     cfg = get_cfg()
#     # disable cuda at the moment
#     cfg.MODEL.DEVICE = args.model_device
#     cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)
#     # only keep the top 5 instances
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
#     model = DefaultPredictor(cfg)
#     return model, cfg


def probe_video_dim(input):
    """Probe the video dimensions."""
    probe = ffmpeg.probe(input)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    return width, height


def get_ffmpeg_input_async(args, start_time: int, duration: int, framerate: int):
    """Initialize the ffmpeg stream."""
    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-video-to-numpy-array
    # https://stackoverflow.com/questions/63623398/read-one-16bits-video-frame-at-a-time-with-ffmpeg-python
    input = ffmpeg.input(args.input).filter("fps", fps=framerate, round="up")
    if duration > 0 or start_time > 0:
        input = input.trim(start_frame=start_time * framerate, duration=duration)
    return input.output("pipe:", format="rawvideo", pix_fmt="rgb24").run_async(
        pipe_stdout=True
    )


def get_ffmpeg_output_async(args, width: int, height: int, framerate: int):
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
        .output(args.output, pix_fmt="yuv420p", vcodec="libx264")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


class DataPickleWriter:
    """Write a stream of data to a series of pickle files."""

    def __init__(self, output_data_root: str, chunk_size=16 * 30):
        self.output_data_root = Path(output_data_root)
        self.chunk_size = chunk_size
        self.chunk_index = 0
        self.chunk = []

    def write(self, data):
        self.chunk.append(data)
        if len(self.chunk) == self.chunk_size:
            self.flush()

    def _chunk_to_df(self, chunk):
        return pd.DataFrame(
            [
                dict(
                    frame_index=d.frame_index,
                    framerate=d.framerate,
                    pred_boxes=d.instances.pred_boxes,
                    scores=d.instances.scores,
                    pred_classes=d.instances.pred_classes,
                    pred_keypoints=d.instances.pred_keypoints,
                )
                for d in chunk
            ]
        )

    def flush(self):
        self.output_data_root.mkdir(parents=True, exist_ok=True)
        # use 4 digits for the chunk
        chunk_name = f"chunk_{self.chunk_index:04d}.pkl.zstd"
        self._chunk_to_df(self.chunk).to_pickle(self.output_data_root / chunk_name)
        self.chunk_index += 1
        self.chunk = []


@dataclass
class InferenceData:
    frame_index: int
    framerate: int
    instances: object


def main():
    args = parse_args()
    # this is actually the DefaultPredictor
    model, cfg = get_detectron_model(args)
    width, height = probe_video_dim(args.input)

    in_stream = get_ffmpeg_input_async(
        args,
        start_time=args.start_time,
        duration=args.duration,
        framerate=args.framerate,
    )
    out_stream = get_ffmpeg_output_async(args, width, height, framerate=args.framerate)

    # process frame by frame
    if args.output_data_root:
        pickle_writer = DataPickleWriter(args.output_data_root)
        print(f"writing data to {args.output_data_root}")
    else:
        pickle_writer = None
    index = 0
    while True:
        in_bytes = in_stream.stdout.read(width * height * 3)
        if not in_bytes:
            break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        prediction = model(in_frame)
        instances = prediction["instances"].to("gpu")
        # now write out an image to test
        v = Visualizer(
            in_frame[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=1,
        )
        
        out_frame = v.draw_instance_predictions(instances)
        out_stream.stdin.write(
            out_frame.get_image()[:, :, ::-1].astype(np.uint8).tobytes()
        )
        if pickle_writer:
            pickle_writer.write(
                InferenceData(
                    frame_index=index,
                    framerate=args.framerate,
                    instances=instances,
                )
            )
        index += 1

    if pickle_writer:
        pickle_writer.flush()
    out_stream.stdin.close()
    out_stream.wait()
    in_stream.wait()


if __name__ == "__main__":
    main()
