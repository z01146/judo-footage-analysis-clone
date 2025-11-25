import math

import cv2
import cvzone
from tqdm import tqdm
from ultralytics import YOLO

vidcap = cv2.VideoCapture(
    "/home/GPU/tsutar/local_storage/judo-footage-analysis/interim/mat-2-trunc.mp4"
)

device = "cuda"  # cpu, cuda
img = cv2.imread(
    "/home/GPU/tsutar/home_gtl/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/frames/frame137.jpg"
)
save_path = "/home/GPU/tsutar/home_gtl/intro_to_res/"
model_path = (
    "/mnt/cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt"
)
size = (int(vidcap.get(3)), int(vidcap.get(4)))

out_writer = cv2.VideoWriter(
    save_path + "/bb_overlay_v2.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, size
)

model = YOLO(model_path)

frame_idx = 0

classNames = ["player blue", "player white", "referee"]
colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # (b,g,r) somehow ;(

while vidcap.isOpened():
    success, img = vidcap.read()
    frame_idx += 1

    # if frame_idx == 310:
    #     break

    if not success:
        break

    results = model.predict(img, stream=True, verbose=False)

    i = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = box.cls[0]
            # cvzone.cornerRect(img, (x1, y1, w, h), colorR=colors[int(cls)])
            start_pt = (x1, y1)
            end_pt = (x2, y2)
            cv2.rectangle(img, start_pt, end_pt, color=colors[int(cls)], thickness=2)

            conf = math.ceil((box.conf[0] * 100)) / 100

            name = classNames[int(cls)]

            cvzone.putTextRect(
                img,
                f"{name} " f"{conf}",
                (max(0, x1), max(35, y1)),
                scale=1.5,
                thickness=2,
            )

    out_writer.write(img)
