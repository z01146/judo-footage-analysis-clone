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
        **kwargs,
    ):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        self.labels = ["referee", "player_blue", "player_white"]
        self.model = YOLO(model_name)
        self.base_url = base_url
        self.api_token = api_token
        self.model_version = "v8n_v1"

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

        def area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        def label(prediction, image, labels=self.labels):
            # we use the dominant color of the box to determine the label
            # if it's black, it's a referee
            # if its blue, it's a player_blue
            # if its white, it's a player_white
            cropped = np.array(image.crop(prediction.xyxy[0].tolist()))
            # sum across all the pixels into rgba channels
            Z = cropped.reshape((-1, 3))
            # for each pixel, calculate the difference to the three colors
            dist_black = np.linalg.norm(Z - np.array([0, 0, 0]), axis=1)
            # we actually want to be closer to dark blue
            dist_blue = np.linalg.norm(Z - np.array([24, 19, 81]), axis=1)
            dist_white = np.linalg.norm(Z - np.array([255, 255, 255]), axis=1)
            # now find the counts of each color
            scores = np.stack([dist_black, dist_blue, dist_white]).T
            best_index = np.argmin(scores, axis=1)
            # count the most frequent index
            return labels[np.bincount(best_index).argmax()]

        for result in results:
            # only keep the top 3 predictions that have the largest boxes
            biggest_predictions = sorted(
                result.boxes, key=lambda x: area(x.xyxy[0]), reverse=True
            )[:3]

            for i, prediction in enumerate(biggest_predictions):
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
                            "rectanglelabels": [label(prediction, image)],
                        },
                    }
                )
                score += prediction.conf.item()

        result = [
            {
                "result": predictions,
                "score": score / len(predictions) + 1,
                # all predictions will be differentiated by model version
                # "model_version": self.model_version,
            }
        ]
        return result
