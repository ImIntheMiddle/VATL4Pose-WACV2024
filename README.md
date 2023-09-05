
<div align="center">
    <img src=".github/overview.png", width="400">
</div>

## ATL4VSHPE: Active Transfer Learning for Efficient Video-Specific Human Pose Estimation
This is the official implementation of the paper ``Active Transfer Learning for Efficient Video-Specific Human Pose Estimation'' submitted to WACV2024.

## Installation
Following the command below, please install the required libraries
```pip
install -r requirement.txt
```

## Pre-trained Model
We will provide pre-trained models at Google Drive after the decision of WACV Round 2.

## Quick Start
Examples: Video-specific Active Transfer Learning on `PoseTrack21`, using `SimpleBaseline`.

- **Train an initial pose estimator from scratch**
``` bash
./scripts/train.sh ${CONFIG} ${EXP_ID}
```

- **Evaluate the performance of pre-trained model on train/val/test split**
``` bash
./scripts/validate.sh ${CONFIG} ${CHECKPOINT}
```

- **Execute Video-specific Active Transfer Learning on test videos**
``` bash
./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml pretrained_models/fast_res50_256x192.pth ${VIDEO_NAME}
#or
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
#or if you want to use yolox-x as the detector
python scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

- **Summarize the results of video-specific ATL**
``` bash
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

- **(Optional) Visualize the estimated poses on each ATL cycle**
``` bash
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```
## Acknowledgement
This implementation is based on AlphaPose, ALiPy, and VL4Pose.
We deeply appreciate the authors for their open-source codes.

## License
The use of code under this repository follows the MIT License. Please see the LICENSE.txt file for details.
