import os
import json
from judo_footage_analysis.combat_phase.workflow import Workflow

# --------------------------
# CONFIGURE PATHS FOR YOUR MACHINE1`
# --------------------------
VIDEO_FOLDER = r"C:\Users\v5karthi\Desktop\converted_mp4"
JSON_PATH = r"C:\Users\v5karthi\Desktop\judo-footage-analysis-main\data\combat_phase\project.json"
OUTPUT_FOLDER = r"C:\Users\v5karthi\Desktop\judo-footage-analysis-main\data\combat_phase\discrete_v2"

# --------------------------
# Generate JSON from videos
# --------------------------
videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
data = [{"video": os.path.join(VIDEO_FOLDER, v), "annotations": []} for v in videos]

os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True
print(f"Workflow finished. Output saved at: {OUTPUT_FOLDER}")
