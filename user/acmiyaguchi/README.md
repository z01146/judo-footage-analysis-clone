# acmiyaguchi

This is a directory for me to throw in random scripts or docs that may not belong elsewhere related to analysis.

## detectron demo on truncated videos

```bash
time ./user/acmiyaguchi/run_detectron.py \
    --input data/interim/mat-2-trunc.mp4 \
    --output data/interim/mat-2-viz.mp4 \
    --output-data-root data/interim/mat-2-data \
    --framerate 10

real    36m58.280s
user    49m58.235s
sys     0m21.837s

time ./user/acmiyaguchi/run_detectron.py \
    --input data/interim/mat-8-trunc.mp4 \
    --output data/interim/mat-8-viz.mp4 \
    --output-data-root data/interim/mat-8-data \
    --framerate 10

real    29m41.595s
user    42m50.767s
sys     0m18.730s

time ./user/acmiyaguchi/run_detectron.py \
    --input data/interim/mat-3-trunc.mp4 \
    --output data/interim/mat-3-viz.mp4 \
    --output-data-root data/interim/mat-3-data \
    --framerate 10

real    35m40.833s
user    49m28.443s
sys     0m22.649s
```

```bash
python3 -m b2 sync data/interim b2://acm-judo/data/analysis/interim
```
