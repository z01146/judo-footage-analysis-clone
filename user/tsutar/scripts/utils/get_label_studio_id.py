import json
import os


def extract_filename_and_parents(image_url):
    filename = os.path.basename(image_url)
    parent = os.path.basename(os.path.dirname(image_url))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(image_url)))
    return filename, parent, grandparent


def find_matching_ids(json_file, duplicate_files):
    matching_ids = []

    # Read duplicate files
    with open(duplicate_files, "r") as f:
        duplicate_files_list = [line.strip() for line in f]

    # Read JSON file
    with open(json_file, "r") as f:
        json_data = json.load(f)
        for obj in json_data:
            image_url = obj["data"]["image"]
            filename, parent, grandparent = extract_filename_and_parents(image_url)
            for dup_file in duplicate_files_list:
                (
                    dup_filename,
                    dup_parent,
                    dup_grandparent,
                ) = extract_filename_and_parents(dup_file)
                if (
                    filename == dup_filename
                    and parent == dup_parent
                    and grandparent == dup_grandparent
                ):
                    matching_ids.append(obj["id"])

    return matching_ids


def main():
    json_file = "labelStudio.json"
    duplicate_files = "./duplicate_files.txt"

    if os.path.isfile(json_file) and os.path.isfile(duplicate_files):
        matching_ids = find_matching_ids(json_file, duplicate_files)

        # Write matching ids to a new file
        with open("matching_ids.txt", "w") as f:
            for id_ in matching_ids:
                f.write(str(id_) + "\n")
    else:
        print("Please ensure that file.json and duplicate_files.txt exist.")


if __name__ == "__main__":
    main()
