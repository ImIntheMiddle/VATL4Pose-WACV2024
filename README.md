# 🏊[VATL4Pose](https://arxiv.org/abs/2311.05041)🎬
> **Note**
> This is an official implementation of the following two papers from **[IIM, TTI-J](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/index.html).**
> - **Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (WACV2024 main)**
>   - Author: [Hiromu Taketsugu](https://iminthemiddle.github.io/), [Norimichi Ukita](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/index.html)
>   - [Project page](https://iminthemiddle.github.io/VATL4Pose-Page/)
>   - [arXiv](https://arxiv.org/abs/2311.05041)
> - **Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation (MVA2023 Oral)**
>   - Author: [Hiromu Taketsugu](https://iminthemiddle.github.io/), [Norimichi Ukita](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/index.html)
>   - [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10215565)

> **Warning**
> The use of code under this repository follows the MIT License. Please take a look at LICENSE for details.

<div align="center">
    <img src=".github/overview.png", width="960">
</div>

## ☑️TODO
- [x] Release pre-trained models
- [x] Release the paper on arXiv
- [x] Release a project page

## 📑Abstract
***Human Pose (HP) estimation** is actively researched because of its wide range of applications. However, even estimators pre-trained on large datasets may not perform satisfactorily due to a domain gap between the training and test data. To address this issue, we present our approach **combining Active Learning (AL) and Transfer Learning (TL) to adapt HP estimators to individual video domains efficiently.** For efficient learning, our approach quantifies **(i) the estimation uncertainty based on the temporal changes in the estimated heatmaps** and **(ii) the unnaturalness in the estimated full-body HPs.** These quantified criteria are then effectively **combined with the state-of-the-art representativeness criterion** to select uncertain and diverse samples for efficient HP estimator learning. Furthermore, we reconsider the existing Active Transfer Learning (ATL) method to introduce **novel ideas related to the retraining methods and Stopping Criteria (SC).** Experimental results demonstrate that our method enhances learning efficiency and outperforms comparative methods.*

## ⬇️Installation
> **Warning**
> Environment: Python 3.10.7, CUDA 11.3, PyTorch 1.12.1
> 
> Other versions have not been tested.
- Create and activate a virtual environment for this repository.
- Following the command below, please install the required packages:
    ```
    pip install -r requirement.txt
    ```
- Then, you need to setup pycocotools by following:
    ```
    cd data/coco/cocoapi/PythonAPI
    python setup.py install
    ```
    
## 🌐Downloads
- Please download PoseTrack21 and JRDB-Pose, and place them under the `./data` directory.
    - PoseTrack21: https://github.com/anDoer/PoseTrack21
    - JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose
- After downloading, you can prepare annotation files as follows (please specify the mode in each script):

<details><summary>PoseTrack21</summary>

```
python ./data/PoseTrack21/make_new_annotation.py
python ./data/PoseTrack21/integrate_new_annotation.py
```
</details>

<details><summary>JRDB-Pose</summary>
    
```
python ./data/jrdb-pose/make_new_annotation.py
python ./data/jrdb-pose/integrate_new_annotation.py
```
</details>

- You can download pretrained models of Human Pose Estimator (HRNet, FastPose and SimpleBaseline) and our Wholebody Auto-Encoder from Releases ``Pretrained models''.
    - Unzip and place the ``pretrained_models'' directory under the root directory of the repository.

## 🚀Quick Start
- Make sure you are in the root directory.
- You can execute **VATL (Video-specific Active Transfer Learning)** by following commands.

<details><summary><bold>VATL on PoseTrack21 using SimpleBaseline</bold></summary>
    
1. **(Optional) Train an initial pose estimator from scratch**
    ```
    python ./scripts/posetrack_train.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
2. **(Optional) Evaluate the performance of the pre-trained model on train/val/test split**
    ```
    python ./scripts/poseestimatoreval.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
3. **(Optional) Pre-train the AutoEncoder for WPU (Whole-body Pose Unnaturalness)**
    ```
    python ./scripts/wholebodyAE_train --dataset_type Posetrack21
    ```
4. **Execute Video-specific Active Transfer Learning on test videos**

    > **Warning**
    > Please specify the detailed settings in the shell script if you like.
    ```
    bash ./scripts/run_active_learning.sh ${GPU_ID}
    ```
5. **Evaluate the results of video-specific ATL**

    > **Warning**
    > Please specify the results to summarize in the Python script.
    ```
    python ./scripts/detailed_result.py
    ```
6. **(Optional) Visualize the estimated poses on each ATL cycle**

    > **Warning**
    > Please specify the results to summarize in the Python script.
    ```
    python ./scripts/visualize_result.py
    ```
</details>

## ✍️Citation
**If you found this code useful, please consider citing our work :D**

<details><summary>WACV2024</summary>

```
@InProceedings{VATL4Pose_WACV24,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Active Transfer Learning for Efficient Video-Specific Human Pose Estimation},
  booktitle    = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year         = {2024}}
```
</details>

<details><summary>MVA2023</summary>

```
@InProceedings{VATL4Pose_MVA23,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation}, 
  booktitle    = {2023 18th International Conference on Machine Vision and Applications (MVA)}, 
  year         = {2023}}
```
</details>

## 🤗Acknowledgement
This implementation is based on AlphaPose, ALiPy, DeepAL+, and VL4Pose.
We deeply appreciate the authors for their open-source codes.
- AlphaPose: https://github.com/MVIG-SJTU/AlphaPose
- ALiPy: https://github.com/NUAA-AL/ALiPy
- DeepAL+: https://github.com/SineZHAN/deepALplus
- VL4Pose: https://github.com/meghshukla/ActiveLearningForHumanPose

## 🤝Contributing
If you'd like to contribute, you can open an issue on this repository.

All contributions are welcome! All content in this repository is licensed under the MIT license.
