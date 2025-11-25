from ultralytics import YOLO

# import torch

# Load a model
model = YOLO(
    "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scripts/for_training/runs/classify/train18/weights/best.pt"
)  # load a pretrained model (recommended for training)
# print(torch.cuda.devices())

# Train the model
results = model.train(
    data="/home/GTL/tsutar/intro_to_res/referee_dataset_v2/",
    epochs=100,
    imgsz=640,
    device="0",
    patience=10,
)
