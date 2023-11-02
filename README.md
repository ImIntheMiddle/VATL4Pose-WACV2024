# VATL4Pose
This is the official implementation of the following two papers from IIM, TTI-J.
    - Homepage: https://www.toyota-ti.ac.jp/Lab/Denshi/iim/index.html
- **Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (WACV2024 main)**
    - Project page: Not Yet
    - arXiv: 404
- **Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (MVA2023 Oral)**
    - PDF (IEEE Xplore): https://ieeexplore.ieee.org/abstract/document/10215565

<div align="center">
    <img src=".github/overview.png", width="960">
</div>

## Installation
Following the command below, please install the required libraries:
```pip
install -r requirement.txt
```

Please download PoseTrack21 and JRDB-Pose, and place them under the `./data` directory.
- PoseTrack21: https://github.com/anDoer/PoseTrack21
- JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose

## Pre-trained Model
We will provide pre-trained models at Google Drive soon.

## Quick Start
Examples: Video-specific Active Transfer Learning on `PoseTrack21`, using `SimpleBaseline`.

- **Train an initial pose estimator from scratch**
    ``` python
    ./scripts/posetrack_train.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
- **Evaluate the performance of the pre-trained model on train/val/test split**
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
- **Evaluate the results of video-specific ATL**

    Please specify the results to summarize in the Python script.
    ``` python
    ./scripts/detailed_result.py
    ```
- **(Optional) Visualize the estimated poses on each ATL cycle**

    Please specify the results to summarize in the Python script.
    ``` python
    ./scripts/visualize_result.py
    ```
## Citation
If you found this code useful, please consider citing our work:D
- WACV2024
```
    Coming soon!
```
- MVA2023
```
@INPROCEEDINGS{10215565,
  author={Taketsugu, Hiromu and Ukita, Norimichi},
  booktitle={2023 18th International Conference on Machine Vision and Applications (MVA)}, 
  title={Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation}, 
  year={2023},
  volume={},
  number={}
  pages={1-5},
  doi={10.23919/MVA57639.2023.10215565}}
```

## Acknowledgement
This implementation is based on AlphaPose, ALiPy, and VL4Pose.
We deeply appreciate the authors for their open-source codes.
- AlphaPose: https://github.com/MVIG-SJTU/AlphaPose
- ALiPy: https://github.com/NUAA-AL/ALiPy
- VL4Pose: https://github.com/meghshukla/ActiveLearningForHumanPose

## License
The use of code under this repository follows the MIT License. Please see the LICENSE.txt file for details.
