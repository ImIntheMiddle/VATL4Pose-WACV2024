# MODEL ZOO

## MSCOCO dataset 

| Model                    | Backbone | Detector | Input Size |     AP     | Speed |  Download | Config | Training Log |  
|--------------------------|----------|----------|------------|------------|-------|-----------|--------|--------------|
|[Simple Baseline](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | ResNet50           | YOLOv3 | 256x192            | 70.6        | 2.94 iter/s | [model](https://drive.google.com/open?id=1nxyfUbvWDaaT9eDM7Y31ScSVPlGy6gfw) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | [log](logs/simple_res50_256x192.log) |
|[Fast Pose](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | ResNet50           | YOLOv3 | 256x192            | 72.0        | 3.54 iter/s | [model](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | [log](logs/fast_res50_256x192.log) |
|[HRNet](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | HRNet-W32           | YOLOv3 | 256x192            | 72.5        | 2.13 iter/s | [model](https://drive.google.com/open?id=1i63BPlOnp2vSjIZ7ni4Yp3RCPQwqe922) | [cfg](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | [log](logs/hrnet_w32_256x192.log) |
#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- One TITAN XP is used for speed test, with `batch_size=64` in each iteration.
- Offline human detection results are used in speed test.
- `FastPose` is our own network design. Paper coming soon!