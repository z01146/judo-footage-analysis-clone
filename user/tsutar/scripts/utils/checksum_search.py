import glob
import hashlib
import os


def calculate_checksum(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicate_files(folder1, folder2):
    checksum_dict = {}
    duplicate_files = []

    # Calculate checksums for files in the first folder
    for file in sorted(glob.glob(os.path.join(folder2, "*/*.png"))):
        checksum = calculate_checksum(file)
        checksum_dict.setdefault(checksum, []).append(file)

    # Check for duplicates in the second folder
    for file in sorted(glob.glob(os.path.join(folder1, "*/*/*.png"))):
        checksum = calculate_checksum(file)
        if checksum not in checksum_dict:
            duplicate_files.extend(checksum_dict[checksum])
            # duplicate_files.append(checksum_dict[checksum])

    return duplicate_files


def main():
    folder1 = "/cs-share/pradalier/tmp/judo/data/referee_v2/"
    folder2 = "/cs-share/pradalier/tmp/judo/data/referee_v2_sorted/classes/"
    # folder3 = "/cs-share/pradalier/tmp/judo/data/referee_v2_sorted/referee_retrain/"

    checksum_dict = {}

    # # Calculate checksums for files in the first folder
    # for file in sorted(glob.glob(os.path.join(folder1, "*/*/*.png"))):
    #     checksum = calculate_checksum(file)
    #     checksum_dict[file] = checksum

    # # Save checksums and file paths to a dictionary
    # with open("checksums.txt", "w") as f:
    #     for filepath, checksum in checksum_dict.items():
    #         f.write(f"{checksum}: {filepath}\n")

    # for file in sorted(glob.glob(os.path.join(folder2, "*/*.png"))):
    #     checksum = calculate_checksum(file)
    #     checksum_dict[file] = checksum

    # # Save checksums and file paths to a dictionary
    # with open("checksums.txt", "w") as f:
    #     for filepath, checksum in checksum_dict.items():
    #         f.write(f"{checksum}: {filepath}\n")

    # Find and export duplicate files
    duplicate_files = find_duplicate_files(folder1, folder2)
    with open("duplicate_files.txt", "w") as f:
        for file in duplicate_files:
            f.write(file + "\n")


if __name__ == "__main__":
    main()
