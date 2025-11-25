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
        model_version="v8n_v1",
        **kwargs,
    ):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        self.labels = ["player_blue", "player_white", "referee"]
        self.model = YOLO(model_name)
        self.base_url = base_url
        self.api_token = api_token
        self.model_version = model_version

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
        results = self.model.predict(image)

        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                # print(prediction)
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.labels[int(prediction.cls.item())]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()

        result = [
            {
                "result": predictions,
                "score": score / (i + 1),
                # all predictions will be differentiated by model version
                "model_version": self.model_version,
            }
        ]
        return result
