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

