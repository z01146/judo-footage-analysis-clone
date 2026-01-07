# judo-footage-analysis

This repository is work supporting "Semi-Supervised Extraction and Analysis of Judo Combat Phases
from Recorded Live-Streamed Tournament Footage".
The goal of the project is to automate live-stream recording segmentation into individual matches, extract combat phases from matches, and to gather statistics at the tournament level.

This project was done as part of CS8813 Introduction to Research at Georgia Tech Europe during the Spring 2024 semester.

## quickstart

Checkout the repo and install any dependencies you may need to a virtual environment:

```bash
git checkout ${repo}
cd ${repo_name}

python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Install any of the relevant tools for running workflows:

- ffmpeg
- b2-tools
- google-cloud-sdk

### running a workflow

Most of the data processing workflows are written as [luigi](https://github.com/spotify/luigi) scripts under the [judo_footage_analysis/workflow](./judo_footage_analysis/workflow) module.
These can be run as follows:

```bash
# in a terminal session
luigid

# in a separate session
python -m judo_footage_analysis.workflow.{module_name}
```

You can watch the progress of a job in the terminal or from the luigi web-ui at http://localhost:8082.


## Conversion of the Provided MKV File to Mp4

This project uses FFmpeg, provided through the Python Package
imageio-ffmpeg, bundles a local FFmpeg binary inside the virtual environment
This allows video processing without installing the python package on your OS


### Activating the Virtual Environment
*If your virtual environment is already active skip this step*
```bash
.\.venv\Scripts\Activate
```
Once active, the prompt should look like:

`(.venv) PS C:\path\to\project`

### Locating the FFmpeg Binary inside the venv

The following command prints the full path for FFmpeg executable to access it
```bash
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```
### Converting MKV to MP4 Using the venv FFmpeg

Use the full FFmpeg path obtained above:

*Example:*
```
& "C:\Users\<username>\judo-footage-analysis-main\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe" -i "C:\Users\<username>\Desktop\OntarioOpen_Mat5_Sat2025.mkv" "C:\Users\<username>\Desktop\OntarioOpen_Mat5_Sat2025.mp4"
```
- `-i` in the command specifies the MKV file being inputted
- The last argument specifies the output MP4 file

Once you have the full FFmpeg path from Step 2, you can use it to run a conversion command. 
Because PowerShell cannot run executables with spaces in their path directly, we use `&` to invoke the executable.

## Setting the Output Folder for the Frames
This project allows extraction of frames from videos for further analysis. You can configure where these frames are saved, either using a default folder or specifying a custom location.
By default, frames are saved in folder relative to the input video "output_frames"

*You can specify a custom folder in your script or workflow. For example:*
```bash
from judo_footage_analysis.frame_extraction import extract_frames

video_path = "path/to/video.mp4"
output_folder = "path/to/output_frames"

extract_frames(video_path, output_folder)
```
- `output_folder` is the path where all extracted frames will be saved
- The folder will be automatically created if it does not exist

If your using the workflow, use the following command:
```
python -m judo_footage_analysis.workflow.extract_frames \
    --video "videos/match1.mp4" \
    --output_folder "frames/match1_frames"
```
This allows you to control where the output of the frames go.

## Automatic FLV to MP4 Conversion (FLV File Provided by Eugene)
The project includes a Python script to automatically convert FLV files to MP4. The script lives inside the repository, so you can run it directly from your project terminal

### Running the Script
Open your terminal and turn on your virtual environment
```
.\.venv\Scripts\Activate
```
Call the conversion script
```
python scripts\convert_flv_auto.py
```
The script will do the following:
- Scan the folder specified inside the script (specified for Desktop) for .flv files
- Convert them to MP4
- Saves converted files in a `converted_mp4` folder in the same location

*NOTE: This script will only pick up your FLV Video File inside your Desktop and it should be outside the folder directly on the Desktop*

### Video Segmentation of the Converted MP4 File
After converting to MP4, you can segment a long video into individual Judo matches using the `truncate_videos.py` workflow
```bash
python -m judo_footage_analysis.workflow.truncate_videos `
    --input-root-path "C:\Users\v5karthi\Desktop\converted_mp4" `
    --output-root-path "C:\Users\v5karthi\Desktop\segmented_matches" `
    --output-prefix "match_" `
    --duration 600 `
    --num-workers 1
```

What each variable means:
- `--input-root-path` – folder containing MP4 files to segment
- `--output-root-path` – folder where segmented matches will be saved

Output will be MP4 files in the specified output folder ready for frame extraction.

## Frame Extraction
Frames can be extracted from segmented videos for further analysis of each fight:
```bash
from judo_footage_analysis.frame_extraction import extract_frames

video_path = "path/to/match.mp4"
output_folder = "path/to/frames"

extract_frames(video_path, output_folder)
```

or you can use the workflow:
```bash
python -m judo_footage_analysis.workflow.extract_frames \
    --video "videos/match1.mp4" \
    --output_folder "frames/match1_frames"
```
## Generating the Project JSON (Required for Later Workflows)
Some workflows in this repository require a JSON file listing all videos to be processed.
To simplify this step, a script is included to automatically create this JSON file.

A script named `generate_video_json.py` is located under the `scripts/` folder. It scans a folder containing your MP4 files and generates a JSON listing each video.
*Virtual Envoirnment should be active at all times throughout this code unless specified*

**Run the JSON generator:**
```bash
python scripts\generate_video_json.py
```
**This script creates a JSON file at:**
```bash
judo-footage-analysis-main/data/combat_phase/project.json
```
## Combat Phase Extraction
### Install dependencies

```bash
pip install -r requirements.txt
pip install imageio[ffmpeg]
````

After preprocessing, run the phase classifier:
```bash
python -m judo_footage_analysis.workflow.extract_combat_phases \
    --project-json data/combat_phase/project.json \
    --output-dir data/combat_phase/results
```
This workflow does the following:
- Frame loading
- Pose or motion feature extraction (depending on model)
- Semi-supervised classification of combat phases

## Video Segmentation
After converting your livestream recording to MP4, you can segment the long file into individual Judo matches using the `truncate_videos` Luigi workflow.

This workflow cuts the video into fixed-length segments (10 minutes each) and saves them to an output folder.

*NOTE: Segmenting very large recordings (20–30 GB+) can take several hours, especially on laptops. Segments will appear one by one in your `segmented_matches/` folder as they finish.*

**Running the Video Segmentation Workflow**
In your activated virtual environment
```bash
python -m judo_footage_analysis.workflow.truncate_videos \
    --input-root-path "C:\Users\<username>\Desktop\converted_mp4" \
    --output-root-path "C:\Users\<username>\Desktop\segmented_matches" \
    --output-prefix "match_" \
    --duration 600 \
    --num-workers 1
```
What it means:
- `--input-root-path` - Folder containing the input MP4 file to segment
- `--output-root-path` - Folder where segmented match clips will be saved
- `--output-prefix` - Prefix applied to each segment file name
- `--duration` - Length (in seconds) of each output clip (example: `600` = 10 minutes)
- `--num-workers` - Number of parallel workers; keep at 1 on most laptops

  **Example Output Files**
```bash
match_0001.mp4
match_0002.mp4
match_0003.mp4
```
They will keep on appearing until the segmenting is complete. If you want longer commands edit the `--duration` variable to edit the time.
Segments will appear one by one in your `segmented_matches/` folder as they finish and with it finishing it'll issue a _SUCCESS file which'll confirm that the segmenting is done.


## Combat Phase Extraction (Machine Learning)

This workflow uses a YOLOv8 object detection model to analyze judo matches and classify combat into Tachi-waza (standing) or Ne-waza (groundwork) based on athlete bounding box statistics.

**Generating the Project JSON**

Before running the ML workflow, you must generate a project manifest. This script scans your segmented matches and creates a "map" for the AI.

```bash
# Run the JSON generator
python scripts/generate_combat_json.py
```
- Input Folder: Scans `Desktop/segmented_matches` by default
- Output File: Saves the manifest to `data/combat_phase/project.json`

**Running the Extraction Workflow**
Ensure the `luigid` scheduler is running in a separate terminal window. Then execute the extraction using the following commands:

```bash
# Set PYTHONPATH so Python recognizes the local project modules
$env:PYTHONPATH = "."

# Run the ML Workflow Task
python -m judo_footage_analysis.workflow.extract_combat_phases ExtractCombatPhases `
    --project-json "data/combat_phase/project.json" `
    --output-dir "data/combat_phase/results"
```
*Key Features:*
- Automatically downloads the yolov8n.pt weights on the first run
- Includes a built-in fix for Windows `CERTIFICATE_VERIFY_FAILED` errors during model downloads
- Handles various `JSON` keys (e.g., `video`, `path`, `video_path`) to prevent `KeyError` crashes


