import os
import shutil

# Define your dataset directory and class mapping
dataset_dir = "/cs-share/pradalier/tmp/judo/data/referee_v2_sorted/classes/"
output_dir = "/cs-share/pradalier/tmp/judo/data/referee_v2_sorted/dataset/"
class_map = {
    "match_stop": 0,
    "point": 1,
    "half_point": 2,
    "other": 3,
}  # Update with your classes and corresponding indices

# Create directories for images and labels
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)


# Function to create label files
def create_label_file(image_name, class_index):
    label_filename = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_filename, "w") as f:
        f.write(str(class_index))


# Iterate through class folders
for class_name in class_map:
    class_dir = os.path.join(dataset_dir, class_name)
    # Iterate through images in each class folder
    for filename in os.listdir(class_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(class_dir, filename)
            # Copy image to images directory
            shutil.copy(image_path, images_dir)
            # Create label file
            create_label_file(filename, class_map[class_name])

# Write class mapping to file
class_map_file = os.path.join(output_dir, "class_map.txt")
with open(class_map_file, "w") as f:
    for class_name, class_index in class_map.items():
        f.write(f"{class_name}\n")
