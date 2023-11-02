# VATL4Pose
> **Note**
> This is an official implementation of the following two papers from IIM, TTI-J (https://www.toyota-ti.ac.jp/Lab/Denshi/iim/index.html).
> - **Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (WACV2024 main)**
>   - Project page: Not Yet
>   - arXiv: 404
> - **Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (MVA2023 Oral)**
>   - PDF (IEEE Xplore): https://ieeexplore.ieee.org/abstract/document/10215565

> **Warning**
> The use of code under this repository follows the MIT License. Please see the LICENSE.txt for details.

<div align="center">
    <img src=".github/overview.png", width="960">
</div>

## Installation
- Create and activate a virtual environment.
- Following the command below, please install the required packages:
    ```pip
    install -r requirement.txt
    ```
    
## Data Preparation
- Please download PoseTrack21 and JRDB-Pose, and place them under the `./data` directory.
    - PoseTrack21: https://github.com/anDoer/PoseTrack21
    - JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose

## Pre-trained Model
> **Note**
> We will provide pre-trained models on Google Drive soon.

## Quick Start
You can execute VATL (Video-specific Active Transfer Learning) by following.
<details><summary><bold>Example: ATL on PoseTrack21 using SimpleBaseline as a pose estimator.</bold></summary>

1. **(Optional) Train an initial pose estimator from scratch**
    ``` python
    ./scripts/posetrack_train.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
2. **(Optional) Evaluate the performance of the pre-trained model on train/val/test split**
    ``` python
    ./scripts/poseestimatoreval.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
3. **(Optional) Pre-train the AutoEncoder for WPU (Whole-body Pose Unnaturalness)**
    ``` python
    ./scripts/wholebodyAE_train --dataset_type Posetrack21
    ```
4. **Execute Video-specific Active Transfer Learning on test videos**

    Please specify the detailed settings in the shell script if you like.
    ``` bash
    ./scripts/run_active_learning.sh ${GPU_ID}
    ```
5. **Evaluate the results of video-specific ATL**

    Please specify the results to summarize in the Python script.
    ``` python
    ./scripts/detailed_result.py
    ```
6. **(Optional) Visualize the estimated poses on each ATL cycle**

    Please specify the results to summarize in the Python script.
    ``` python
    ./scripts/visualize_result.py
    ```
</details>

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
