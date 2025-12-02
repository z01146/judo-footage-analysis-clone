# This will server as a small snippet to test if FFMPEG works for the Segmentaiton
from pathlib import Path
import ffmpeg
import imageio_ffmpeg as iio

# Path to the FFmpeg binary
FFMPEG_PATH = Path(iio.get_ffmpeg_exe())

# Path to your test video (pick one small mp4)
input_video = Path(r"C:\Users\v5karthi\Desktop\Converted_mp4\rec_2024-10-20_06_54_42.mp4")
output_clip = Path(r"C:\Users\v5karthi\Desktop\segmented_matches\test_clip.mp4")

# Make sure output directory exists
output_clip.parent.mkdir(parents=True, exist_ok=True)

try:
    # Probe the video (check info)
    info = ffmpeg.probe(str(input_video), cmd=str(FFMPEG_PATH))
    print("Video info:", info["format"]["duration"], "seconds")

    # Create a 10-second clip starting at 0s
    (
        ffmpeg
        .input(str(input_video), ss=0, t=10)
        .output(str(output_clip), format="mp4")
        .run(overwrite_output=True, cmd=str(FFMPEG_PATH))
    )
    print("Clip successfully created at", output_clip)

except ffmpeg.Error as e:
    print("FFmpeg error:")
    print(e.stderr.decode())
