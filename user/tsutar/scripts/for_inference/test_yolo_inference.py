import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import cv2
import luigi
import numpy as np
from ultralytics import YOLO

model_name = (
    "/cs-share/pradalier/tmp/judo/models/referee_pose/v2/weights/09-04-2024-best.pt"
)
yolo_original = (
    "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/yolov8n.pt"
)
model = YOLO(model_name)

input_path = "/cs-share/pradalier/tmp/judo/data/referee_v2/mat_01/0000/0398_00.png"
img = cv2.imread(input_path)
w, h, _ = img.shape

results = model(input_path, verbose=False, device="cpu")

classname = ["half_point", "penalty", "point"]

result_dict = []
# Process results list
for result in results:
    ind = result.probs.top1
    pose = classname[ind]
    score = float(result.probs.top1conf.numpy())
    print(ind)
    print(pose)
    print(score)
    # cv2.putText(img, pose, (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
