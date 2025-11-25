import os
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from judo_footage_analysis.utils import ensure_path

model_name = "/cs-share/pradalier/tmp/judo/models/entity_detection/v4/weights/best.pt"
txt_file = "/cs-share/pradalier/tmp/judo/data/referee_v2_sorted/referee_retrain/original_path_2.txt"
out = "/cs-share/pradalier/tmp/judo/data/referee_v2/retrain_fix/"
frames_root = "/cs-share/pradalier/tmp/judo/frames/"

model = YOLO(model_name)


with open(txt_file, "r") as f:
    for line in tqdm(f):
        filepath = os.path.join(frames_root, line.strip())
        img = cv2.imread(filepath)

        results = model.predict(
            img,
            save=False,
            conf=0.2,
            iou=0.5,
            verbose=True,
            stream=False,
            device="cpu",
        )

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = box.cls[0]

                if not int(cls) == 2:
                    continue

                referee = img[y1 : y1 + h, x1 : x1 + w]
                filename = ensure_path(out) / f"{Path(filepath).stem}_{i:02d}.png"
                cv2.imwrite(filename.as_posix(), referee)
