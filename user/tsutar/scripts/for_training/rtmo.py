#!/usr/bin/env python3
import cv2

vidcap = cv2.VideoCapture(
    "/home/GPU/tsutar/local_storage/judo-footage-analysis/interim/mat-2-trunc.mp4"
)
# success,image = vidcap.read()
# count = 0
# while success and count <= 1000:
#   if count >=100 and count <= 200:
#     cv2.imwrite("./frames/frame%d.jpg" % count, image)     # save frame as JPEG file
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

#!export DISPLAY=:0x
import cv2
from rtmlib import Body, Wholebody, draw_skeleton

device = "cpu"  # cpu, cuda
backend = "onnxruntime"  # opencv, onnxruntime, openvino
img = cv2.imread(
    "/home/GPU/tsutar/home_gtl/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/frames/frame137.jpg"
)
save_path = "/home/GPU/tsutar/home_gtl/intro_to_res/"
size = (int(vidcap.get(3)), int(vidcap.get(4)))

result = cv2.VideoWriter(
    save_path + "/mat-2-pose.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 10, size
)

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

# wholebody = Wholebody(to_openpose=openpose_skeleton,
#                       mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
#                       backend=backend, device=device)
body = Body(
    to_openpose=openpose_skeleton,
    mode="balanced",  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
    backend=backend,
    device=device,
)

# keypoints, scores = wholebody(img)
frame_idx = 0

while vidcap.isOpened():
    success, frame = vidcap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(
        img_show, keypoints, scores, openpose_skeleton=openpose_skeleton, kpt_thr=0.43
    )

    result.write(img_show)
    # ------Below code for displaying the frames
    # img_show = cv2.resize(img_show, (960, 540))
    # cv2.imshow('img', img_show)
    # cv2.waitKey(1)


# -----For WholeBody API-----------
# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

# img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.2)


# cv2.imshow('img', img_show)
# cv2.waitKey()
