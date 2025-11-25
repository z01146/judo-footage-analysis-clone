## Dataset information

Images used for training YOLO for object detection:
 - first training instance: 697
 - second training instance: 1296

<!-- Steps to serve local files over url. -->


# generate a manifest
```
./scripts/generate_folder_manifest.sh \
    /cs-share/pradalier/tmp/judo \
    '*/referee_v2/*.png' \
    /cs-share/pradalier/tmp/judo/referee_files.txt
```
# start up nginx
```
./scripts/serve_local_files.sh \
    /cs-share/pradalier/tmp/judo

```
<!-- Information about scripts -->
## For training the model
All training scripts are under '/user/tsutar/for_training/'

## For model inference
All training scripts are under '/user/tsutar/for_inference/'

## Utility functions
Scripts required to do utility functions like generating, splitting and cleaning dataset, checksum matching, referee image extraction are under '/user/tsutar/scripts/utils/'


<!-- Setting-up label-studio backend -->
## pre-annotation and active labeling

```bash
python -m user.tsutar.label_studio_backend.referee_pose._wsgi \
    --model-dir /tmp/model_v2 \
    --debug \
    --api-token=...

python -m user.tsutar.label_studio_backend.referee_pose._wsgi \
    --model-dir=/tmp/model_v2 \
    --model-name="/cs-share/pradalier/tmp/judo/models/referee_pose/v2/weights/2024-04-10-best.pt"\
    --debug \
    --api-token=...

python -m user.tsutar.label_studio_backend.entity_detection._wsgi \
    --model-dir=/tmp/model_v2 \
    --model-name="/cs-share/pradalier/tmp/judo/models/entity_detection/v3/weights/best.pt"\
    --api-token=...
```
