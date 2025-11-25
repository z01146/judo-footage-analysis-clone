import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"/usr/bin/tesseract"  # TODO: Change this to the Tessaract path on the system
)


def is_timer(input_string):
    # Remove leading and trailing whitespaces
    input_string = input_string.strip()

    # Extract the first line
    input_string = input_string.split("\n")[0].strip()

    # Define the regular expression pattern for the timer format
    pattern = re.compile(r"^\d{1,2}\s*:\s*\d{2}$")

    # Check if the input string matches the pattern
    if not pattern.match(input_string):
        return (False, None, None)

    # Split the string into minutes and seconds
    minutes, seconds = map(int, input_string.split(":"))

    # print(f"minutes: {minutes}, seconds: {seconds}")

    # Check if seconds are within the valid range (0 to 59)
    if 0 <= seconds <= 59:
        return (True, minutes, seconds)
    else:
        return (False, None, None)


def extract_timer_from_folder(folder_path, json_path, roi_coordinates):
    # Extract last folder in folder_path
    last_folder_path = folder_path.split("/")[-1]
    mat_folder_path = folder_path.split("/")[-2]

    # Create the json folder if it doesn't exist
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    # Create the json file to store the results if it doesn't exist
    json_filename = os.path.join(
        json_path, "timer_" + mat_folder_path + "_" + last_folder_path + ".json"
    )
    print(f"json_filename: {json_filename}")
    if not os.path.exists(json_filename):
        with open(json_filename, "w") as f:
            f.write("[]")

    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            # print(f"Treating {filename}...")
            image_path = os.path.join(folder_path, filename)
            # Read the image
            img = cv2.imread(image_path)

            # Extract the specified region of interest
            roi = img[
                roi_coordinates[1] : roi_coordinates[3],
                roi_coordinates[0] : roi_coordinates[2],
            ]

            # Convert the region of interest to grayscale for better OCR accuracy
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply thresholding if needed
            _, thresh = cv2.threshold(
                gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Use Tesseract to extract text from the region of interest
            text = pytesseract.image_to_string(thresh)

            # Just for testing
            plt.imshow(thresh, cmap="gray")
            plt.title("Image, text: " + text)
            plt.show()
            print(f"Text: {text}")

            result, minutes, seconds = is_timer(text)

            results.append(
                {
                    "filename": folder_path + filename,
                    "available": result,
                    "minutes": minutes,
                    "seconds": seconds,
                    "raw_text": text,
                }
            )

    # Write the results to the json file
    with open(json_filename, "w") as f:
        json.dump(results, f)

    print(f"Extraction done for {folder_path}!")


def extract_points_from_folder(folder_path, json_path, rois):
    roi_coordinates_player_one = rois[0]
    roi_coordinates_player_two = rois[1]
    roi_coordinates_full_band = rois[2]
    roi_coordinates_to_hide = rois[3]

    # Extract last folder in folder_path
    last_folder_path = folder_path.split("/")[-1]
    mat_folder_path = folder_path.split("/")[-2]

    # Create the json folder if it doesn't exist
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    # Create the json file to store the results if it doesn't exist
    json_filename = os.path.join(
        json_path, "timer_" + mat_folder_path + "_" + last_folder_path + ".json"
    )
    print(f"json_filename: {json_filename}")
    if not os.path.exists(json_filename):
        with open(json_filename, "w") as f:
            f.write("[]")

    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            # print(f"Treating {filename}...")
            image_path = os.path.join(folder_path, filename)
            # Read the image
            img = cv2.imread(image_path)

            # Put all black pixels for the region to hide
            img[
                roi_coordinates_to_hide[1] : roi_coordinates_to_hide[3],
                roi_coordinates_to_hide[0] : roi_coordinates_to_hide[2],
            ] = 0

            # Invert the colors for the region player one
            img[
                roi_coordinates_player_one[1] : roi_coordinates_player_one[3],
                roi_coordinates_player_one[0] : roi_coordinates_player_one[2],
            ] = cv2.bitwise_not(
                img[
                    roi_coordinates_player_one[1] : roi_coordinates_player_one[3],
                    roi_coordinates_player_one[0] : roi_coordinates_player_one[2],
                ]
            )

            # Extract the specified region of interest
            img_roi = img[
                roi_coordinates_full_band[1] : roi_coordinates_full_band[3],
                roi_coordinates_full_band[0] : roi_coordinates_full_band[2],
            ]

            # Convert the region of interest to grayscale for better OCR accuracy
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

            # Apply thresholding if needed
            _, thresh = cv2.threshold(
                gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Use Tesseract to extract text from the region of interest
            text = pytesseract.image_to_string(thresh)

            # Just for testing
            plt.imshow(thresh, cmap="gray")
            plt.title("Image, text: " + text)
            plt.show()
            print(f"Text: {text}")

            result, minutes, seconds = is_timer(text)

            results.append(
                {
                    "filename": folder_path + filename,
                    "available": result,
                    "minutes": minutes,
                    "seconds": seconds,
                    "raw_text": text,
                }
            )

    # Write the results to the json file
    with open(json_filename, "w") as f:
        json.dump(results, f)

    print(f"Extraction done for {folder_path}!")


def extract_timer_from_video(video_path, json_path, roi_coordinates):
    # Extract the name of the video
    print(f"video_path: {video_path}")
    video_name = video_path.split("/")[-1]

    # Create the json folder if it doesn't exist
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    # Create the json file to store the results if it doesn't exist
    json_filename = os.path.join(json_path, "timer_" + video_name + ".json")
    print(f"json_filename: {json_filename}")
    if not os.path.exists(json_filename):
        with open(json_filename, "w") as f:
            f.write("[]")

    results = []

    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Batch size
    batch_size = 1000

    # Loop through the frames and write the results to the json file in batches
    for i in range(0, total_frames, batch_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # print(f"i: {i}")
        for j in range(i, min(i + batch_size, total_frames)):
            ret, frame = cap.read()

            if ret:
                # Extract the specified region of interest
                roi = frame[
                    roi_coordinates[1] : roi_coordinates[3],
                    roi_coordinates[0] : roi_coordinates[2],
                ]

                # Convert the region of interest to grayscale for better OCR accuracy
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Apply thresholding if needed
                _, thresh = cv2.threshold(
                    gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # Use Tesseract to extract text from the region of interest
                text = pytesseract.image_to_string(thresh)

                # Just for testing
                # plt.imshow(thresh, cmap='gray')
                # plt.title('Image, text: ' + text)
                # plt.show()
                # print(f"Text: {text}")

                result, minutes, seconds = is_timer(text)

                results.append(
                    {
                        "frame_number": j,
                        "available": result,
                        "minutes": minutes,
                        "seconds": seconds,
                        "raw_text": text,
                        "filename": video_path,
                    }
                )

        # Write the results to the json file
        with open(json_filename, "w") as f:
            json.dump(results, f)

    print(f"Extraction done for {video_path}!")


def parallelize_extraction_timer(frame_folders, json_folder, roi_coordinates):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        for frame_folder in frame_folders:
            futures.append(
                executor.submit(
                    extract_timer_from_folder,
                    frame_folder,
                    json_folder,
                    roi_coordinates,
                )
            )

        # Wait for all tasks to complete
        for future in futures:
            future.result()


def parallelize_extraction_points(frame_folders, json_folder, rois):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        for frame_folder in frame_folders:
            futures.append(
                executor.submit(
                    extract_points_from_folder, frame_folder, json_folder, rois
                )
            )

        # Wait for all tasks to complete
        for future in futures:
            future.result()


def parallelize_extraction_timer_video(video_folders, json_folder, roi_coordinates):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        print(f"video_folders: {video_folders}")
        for video_folder in video_folders:
            print(f"video_folder: {video_folder}")
            futures.append(
                executor.submit(
                    extract_timer_from_video, video_folder, json_folder, roi_coordinates
                )
            )

        # Wait for all tasks to complete
        for future in futures:
            future.result()
