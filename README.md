
<div align="center">
    <img src="docs/logo.jpg", width="400">
</div>

## ATL4VSHPE: Active Transfer Learning for Efficient Video-Specific Human Pose Estimation
This is the official implementation of the paper ``Active Transfer Learning for Efficient Video-Specific Human Pose Estimation'' submitted to WACV2024.

## Installation


## Pre-trained Model
We will provide pre-trained models at Google Drive after the decision of WACV Round 2.

## Quick Start

- **Training**: Train from scratch
``` bash
./scripts/train.sh ${CONFIG} ${EXP_ID}
```

- **Validation**: Validate your model on MSCOCO val2017
``` bash
./scripts/validate.sh ${CONFIG} ${CHECKPOINT}
```

Examples:

Demo using `FastPose` model.
``` bash
./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml pretrained_models/fast_res50_256x192.pth ${VIDEO_NAME}
#or
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
#or if you want to use yolox-x as the detector
python scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

Train `FastPose` on mscoco dataset.
``` bash
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

## Acknowledgement
This implementation is based on AlphaPose, ALiPy, and VL4Pose.
We deeply appreciate the authors for their open-source codes.

## License
The use of code under this repository follows the MIT License. Please see the LICENSE.txt file for details
