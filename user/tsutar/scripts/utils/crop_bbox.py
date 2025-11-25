import cv2
from ultralytics import YOLO

input_img = "/cs-share/pradalier/tmp/judo/frames/mat_01/0001/0437.jpg"
model_wght = "/cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt"
output_path = "/cs-share/pradalier/tmp/judo/"
model = YOLO(model_wght)


results = model.predict(
    input_img,
    save=False,
    conf=0.2,
    iou=0.5,
    verbose=False,
)

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

        if int(cls) == 2:
            img = cv2.imread(input_img)
            referee = img[y1 : y1 + h, x1 : x1 + w]
            cv2.imwrite(output_path + "referee.jpg", referee)
