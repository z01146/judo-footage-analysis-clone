import subprocess
from pathlib import Path
import sys
import imageio_ffmpeg

# === CONFIGURATION ===
# Change this to the folder where your FLV files are
FLV_FOLDER = Path.home() / "Desktop"
OUTPUT_FOLDER = FLV_FOLDER / "converted_mp4"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# Get the FFmpeg executable from the venv
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

# Loop through all FLV files in the folder
flv_files = list(FLV_FOLDER.glob("*.flv"))

if not flv_files:
    print(f"No FLV files found in {FLV_FOLDER}")
    sys.exit()

print(f"Found {len(flv_files)} FLV file(s) in {FLV_FOLDER}")
print(f"Converted files will be saved in {OUTPUT_FOLDER}\n")

for flv_file in flv_files:
    output_file = OUTPUT_FOLDER / (flv_file.stem + ".mp4")
    print(f"Converting: {flv_file.name} → {output_file.name}")

    cmd = [
        ffmpeg_path,
        "-i", str(flv_file),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_file)
    ]

    subprocess.run(cmd, check=True)
    print(f"Finished: {output_file.name}\n")

print("All conversions complete!")
