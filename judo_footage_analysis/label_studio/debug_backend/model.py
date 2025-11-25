from label_studio_ml.model import LabelStudioMLBase


class DebugModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super().__init__(**kwargs)

    def predict(self, tasks, **kwargs):
        return [
            {
                "result": [
                    {
                        "id": "0",
                        "from_name": "foo",
                        "to_name": "bar",
                        "type": "rectanglelabels",
                        "score": 0.5,
                        "original_width": 100,
                        "original_height": 100,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": 10,
                            "y": 10,
                            "width": 10,
                            "height": 10,
                            "rectanglelabels": ["player"],
                        },
                    }
                ],
                "score": 0.5,
            }
        ]
