# notes

## footage links

### 2023 President's Cup

1. <https://www.youtube.com/watch?v=uPhXtW_f_AE>
2. <https://www.youtube.com/watch?v=Nh_cb1RNV9o>
3. <https://www.youtube.com/watch?v=95paesJw7pk>
4. <https://www.youtube.com/watch?v=xVDw2bhFXgk>
5. <https://www.youtube.com/watch?v=2hYzoJ8HSkk>
6. <https://www.youtube.com/watch?v=B38xef6cHHk>
7. <https://www.youtube.com/watch?v=mDFtwQVP9GM>
8. <https://www.youtube.com/watch?v=6rZvqhUaxOE>
   - The only footage without smoothcomp overlay for match statistics
9. <https://www.youtube.com/watch?v=pE_mDPF0gwI>
10. <https://www.youtube.com/watch?v=OwhJQFx27YM>

## downloading stream data

I create a n2d instance with local-ssd (375gb) to download the streamed videos and upload them into a cloud bucket.
I focus on a subset of videos, in particular mat 1 and mat 8.
The former has the [smoothcomp overlay](https://smoothcomp.com/en), which can be used to segment the videos and extract match information from the competition.
The latter does not have an overlay (likely due to user error), and can be used to test algorithms on a raw camera feed.

Install [yt-dlp](https://github.com/yt-dlp/yt-dlp):

```bash
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp
```

Then download the videos to the nvme drive mounted on `/mnt/data`.

```bash
cd /mnt/data

# just mat 1 and mat 8 to test out the functionality
yt-dlp \
    --concurrent-fragments 2 \
    https://www.youtube.com/watch?v=uPhXtW_f_AE \
    https://www.youtube.com/watch?v=6rZvqhUaxOE

# download the rest, this will take a while
yt-dlp \
    --concurrent-fragments 2 \
    https://www.youtube.com/watch?v=Nh_cb1RNV9o \
    https://www.youtube.com/watch?v=95paesJw7pk \
    https://www.youtube.com/watch?v=xVDw2bhFXgk \
    https://www.youtube.com/watch?v=2hYzoJ8HSkk \
    https://www.youtube.com/watch?v=B38xef6cHHk \
    https://www.youtube.com/watch?v=mDFtwQVP9GM \
    https://www.youtube.com/watch?v=pE_mDPF0gwI \
    https://www.youtube.com/watch?v=OwhJQFx27YM
```

Upload the videos.
Install [b2] for uploading the files into a bucket:

```bash
sudo curl -L https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux -o /usr/local/bin/b2
sudo chmod a+rx /usr/local/bin/b2
```

```bash
b2 authorize-account
b2 sync ./ b2://acm-judo/data/yt-dlp
```

## creation of keypoint annotated videos

See [user/acmiyaguchi/README.md](user/acmiyaguchi/README.md).
Here are a few videos that have been annotated with keypoints at 10hz using detectron2:

- <https://www.youtube.com/playlist?list=PLaBtWXB-9VkbHSHyyY-fjAVD7dO1P2PdO>

## extraction of frames

We extract frames from all the downloaded videos to train a model for full frame classification.

```bash
# start up the luigi daemon
luigid

# run the extraction process as a test
python -m judo_footage_analysis.workflow.sample_frames \
    --input-root-path /mnt/students/video_judo \
    --output-root-path /cs-share/pradalier/tmp/judo/frames \
    --duration 20 \
    --batch-size 5 \
    --num-workers 4

# run the extraction process for real
time python -m judo_footage_analysis.workflow.sample_frames \
    --input-root-path /mnt/students/video_judo \
    --output-root-path /cs-share/pradalier/tmp/judo/frames \
    --duration 3600 \
    --batch-size 600 \
    --num-workers 12
```

## extraction of short videos

```bash
python -m judo_footage_analysis.workflow.truncate_videos \
    --input-root-path /mnt/students/video_judo \
    --output-root-path /cs-share/pradalier/tmp/judo/data/clips \
    --duration 30 \
    --num-workers 4
```

Then generate a manifest:

```bash
./scripts/generate_folder_manifest.sh \
    /cs-share/pradalier/tmp/judo \
    '*/clips/*.mp4' \
    /cs-share/pradalier/tmp/judo/data/clips/files.txt
```

## configuring label studio

Follow the instructions from [label studio](https://labelstud.io/guide/install).
We are configuring this on a lab computer, so we implicitly assume network access to the machine.

We note database issues on the NFS home drive when running label-studio:

```bash
  File "/home/GTL/amiyaguc/.local/lib/python3.10/site-packages/django/db/migrations/executor.py", line 91, in migrate
    self.recorder.ensure_schema()
  File "/home/GTL/amiyaguc/.local/lib/python3.10/site-packages/django/db/migrations/recorder.py", line 70, in ensure_schema
    raise MigrationSchemaMissing("Unable to create the django_migrations table (%s)" % exc)
django.db.migrations.exceptions.MigrationSchemaMissing: Unable to create the django_migrations table (database is locked)
```

We have access to the local disk on gtlpc129 on `/data`.

```bash
label-studio \
    --data-dir /data/judo-label \
    --port 8081

# generate a manifest
./scripts/generate_folder_manifest.sh \
    /cs-share/pradalier/tmp/judo \
    '*/frames/*.jpg' \
    /cs-share/pradalier/tmp/judo/frame_files.txt

# start up nginx
./scripts/serve_local_files.sh
```

Here's some handy documentation:

- Keyboard shortcuts: <https://labelstud.io/guide/labeling.html#Use-keyboard-shortcuts>

Now we can ssh ports 8080 so we can access this from any machine with intranet connectivity:

```bash
USER=amiyaguc
ssh -L 8080:localhost:8080 $USER@gtlpc129.georgiatech-metz.fr
```

If you're already on a gtlpc, this simplifies down to:

```bash
ssh -L 8080:localhost:8080 gtlpc129
```

## pre-annotation and active labeling

Create a `.env` file from the `.env.template`.
In particular, set the label-studio api token.

```bash
python -m judo_footage_analysis.label_studio.yolo_entity_backend.wsgi \
    --model-dir /tmp/model \
    --debug

python -m judo_footage_analysis.label_studio.yolo_trained_entity_backend.wsgi \
    --model-dir /tmp/model \
    --model-name /cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt \
    --debug
```

This debugging process sucks, so we've added a bit of testing too.

## Annotated video with full frame inference

Our trained model is not working correctly against a modern version of torch, so we retrain the model.
On a gpu instance (shire):

```bash
yolo classify train \
    data=/mnt/cs-share/pradalier/tmp/judo/fullframe_classification_dataset \
    model=yolov8n-cls.pt \
    verbose=true \
    patience=20 \
    project=/mnt/gpu_storage/judo-footage-analysis/fullframe_classification_models/v2
```

And then copy over the results:

```bash
rsync -a shire:/mnt/gpu_storage/judo-footage-analysis/fullframe_classification_models/v2/ \
    /cs-share/pradalier/tmp/judo/models/fullframe_classification/v2
```

```bash
python -m judo_footage_analysis.workflow.fullframe_inference
```

This should generate some json files that contain the probability of being part of a particular class.
We can apply these back to the frames.

```bash
python -m judo_footage_analysis.workflow.fullframe_overlay
```
