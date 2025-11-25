from io import BytesIO

import numpy as np
import requests
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys
from PIL import Image
from ultralytics import YOLO


class YOLOv8Model(LabelStudioMLBase):
    def __init__(
        self,
        base_url="http://localhost:8080",
        api_token="",
        model_name="yolov8n.pt",
        model_version="pose_v1",
        model_dir="/tmp/model/",
        **kwargs,
    ):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "Choices", "Image"
        )
        self.labels = ["half_point", "match_stop", "other", "point"]
        self.model = YOLO(model_name)
        self.base_url = base_url
        self.api_token = api_token
        self.model_version = model_version
        self.model_dir = model_dir

    def predict(self, tasks, **kwargs):
        """This is where inference happens: model returns
        the list of predictions based on input list of tasks
        """
        task = tasks[0]

        predictions = []
        score = 0

        header = {"Authorization": f"Token {self.api_token}"}
        image = Image.open(
            BytesIO(requests.get(task["data"]["image"], headers=header).content)
        )
        original_width, original_height = image.size
        results = self.model.predict(image, device="cpu")

        i = 0
        for result in results:
            predictions.append(
                {
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "choices",
                    "score": float(result.probs.top1conf.numpy()),
                    "value": {
                        "choices": [self.labels[result.probs.top1]],
                    },
                }
            )

            score += result.probs.top1conf.numpy()

        result = [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model version": self.model_version,
            }
        ]

        return result
