import os

dataset = "/home/GTL/tsutar/intro_to_res/entity-detection-datasets/v2/"
image_path = os.path.join(dataset, "images/")
labels_path = os.path.join(dataset, "labels/")
# train_path = os.path.join(dataset, "train/labels")
# test_path = os.path.join(dataset, "test/labels")
# val_path = os.path.join(dataset, "val/labels")


# List to store file names containing "3" in the first column
matching_files = []
ids = set()

# Iterate through each file in the folder
for filename in os.listdir(labels_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_path, filename)
        with open(file_path, "r") as file:
            for line in file:
                columns = line.split()
                if columns[0] == "3":
                    matching_files.append(filename)

print("file nums: ", len(matching_files))
with open("fix_annotations.txt", "w") as f:
    for file in matching_files:
        f.write(f"{file}\n")

# for i in ids:
#     print(i)
