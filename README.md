## Active Transfer Learning for Efficient Video-Specific Human Pose Estimation
This is the official implementation of the paper ``Active Transfer Learning for Efficient Video-Specific Human Pose Estimation'' submitted to WACV2024.

<div align="center">
    <img src=".github/overview.png", width="960">
</div>

## Installation
Following the command below, please install the required libraries
```pip
install -r requirement.txt
```

Please download PoseTrack21 and JRDB-Pose, and place them under the `./data` directory.

PoseTrack21: https://github.com/anDoer/PoseTrack21

JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose

## Pre-trained Model
We will provide pre-trained models at Google Drive after the decision of WACV Round 2.

## Quick Start
Examples: Video-specific Active Transfer Learning on `PoseTrack21`, using `SimpleBaseline`.

- **Train an initial pose estimator from scratch**
``` python
./scripts/posetrack_train.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
```

- **Evaluate the performance of pre-trained model on train/val/test split**
``` python
./scripts/poseestimatoreval.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
```

- **Pre-train the AutoEncoder for WPU (Whole-body Pose Unnaturalness)**
``` python
./scripts/wholebodyAE_train --dataset_type Posetrack21
```

- **Execute Video-specific Active Transfer Learning on test videos**
Please specify the detailed settings in the shell script if you like.
``` bash
./scripts/run_active_learning.sh ${GPU_ID}
```

- **Summarize the results of video-specific ATL**
Please specify the results to summarize in the python script.
``` python
./scripts/detailed_result.py
```

- **(Optional) Visualize the estimated poses on each ATL cycle**
Please specify the results to summarize in the python script.
``` python
./scripts/visualize_result.py
```
## Acknowledgement
This implementation is based on AlphaPose, ALiPy, and VL4Pose.
We deeply appreciate the authors for their open-source codes.

## License
The use of code under this repository follows the MIT License. Please see the LICENSE.txt file for details.
