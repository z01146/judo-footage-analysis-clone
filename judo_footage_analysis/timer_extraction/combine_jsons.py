import json
import os


def combine_json_files(folder_path, output_file):
    combined_data = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Read each line of the JSON file and append to combined_data
            with open(file_path, "r") as file:
                data = json.load(file)
                combined_data.extend(data)

    # Write the combined_data to the output JSON file
    with open(output_file, "w") as output_file:
        json.dump(combined_data, output_file, indent=2)
