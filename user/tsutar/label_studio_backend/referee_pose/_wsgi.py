import argparse
import logging
import logging.config
import os

from label_studio_ml.api import init_app

from .model import YOLOv8Model


def parse_args():
    parser = argparse.ArgumentParser(description="Label studio")
    parser.add_argument("-p", "--port", type=int, default=9093, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("-d", "--debug", action="store_true", help="Switch debug mode")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "models"),
        help="Directory where models are stored (relative to the project directory)",
    )
    parser.add_argument(
        "--model-name",
        default="yolov8n.pt",
        help="name of the model to use for inference",
    )
    parser.add_argument(
        "--model-version",
        default="pose_v1",
        help="version of the model to use for inference",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate model instance before launching server",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("LABEL_STUDIO_API_URL", "http://localhost:8080"),
        help="Base URL for the API",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.environ.get("LABEL_STUDIO_API_TOKEN"),
        help="API token for the API",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig()
    if args.log_level:
        logging.root.setLevel(args.log_level)

    if args.check:
        YOLOv8Model()

    app = init_app(
        model_class=YOLOv8Model,
        base_url=args.base_url,
        api_token=args.api_token,
        model_dir=args.model_dir,
        model_name=args.model_name,
        model_version=args.model_version,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
