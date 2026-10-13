# VATL4Pose

Official implementation of [Active Transfer Learning for Efficient Video-Specific Human Pose Estimation](https://arxiv.org/abs/2311.05041) (WACV 2024). [Project page](https://iminthemiddle.github.io/VATL4Pose-Page/)

VATL4Pose adapts a pretrained pose estimator to one video. Each round selects the people to annotate with THC, WPU and DUW, and retrains the estimator on the new labels. The code of the paper version is at the tag `wacv2024`.

## Installation

```bash
git clone https://github.com/ImIntheMiddle/VATL4Pose-WACV2024.git
cd VATL4Pose-WACV2024
uv sync --extra cu124        # cu128 for CUDA 12.8 or newer, cpu without a GPU
source .venv/bin/activate
vatl4pose-download-weights --weights-root pretrained_models
vatl4pose-doctor paths.weights_root=pretrained_models
```

## Reproducing the paper

Download [PoseTrack21](https://github.com/anDoer/PoseTrack21) and [JRDB-Pose](https://jrdb.erc.monash.edu/dataset/pose) into `data/`.

```
data/PoseTrack21/posetrack_data/{train,val}/
data/PoseTrack21/images/{train,val}/
data/jrdb-pose/jrdb2022/labels/
data/jrdb-pose/images/image_stitched/
```

```bash
for e in wacv2024_posetrack21 wacv2024_jrdb; do
  vatl4pose-prepare-annotations experiment=$e
  vatl4pose-prepare experiment=$e
  for s in random hp mpe tpc kmeans coreset thc_wpu_duw; do
    vatl4pose-atl experiment=$e strategy=$s run.device=cuda
  done
  vatl4pose-summarize experiment=$e
done
```

`outputs/<experiment>/summary/` holds the ALC of AP@0.6. The paper reports these values (Tables 2 and 4).

| `strategy=` | PoseTrack21 | JRDB-Pose |
|---|---|---|
| `random` | 96.91 | 95.42 |
| `hp` (LC in the paper) | 95.74 | 92.67 |
| `mpe` | 96.11 | 95.76 |
| `tpc` | 96.40 | 93.76 |
| `kmeans` | 97.65 | 96.41 |
| `coreset` | 98.12 | 95.60 |
| `thc_wpu_duw` (ours) | 98.21 | 96.52 |

## Your own videos

Put the frames and a COCO keypoints file with track ids in `data/mine/`. The boxes and track ids come from your detector and tracker.

```
data/mine/images/clip1/000001.jpg
data/mine/annotations/clip1.json
```

```bash
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1]' dataset.keypoints=posetrack21
vatl4pose-atl dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1]' dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  oracle=coco_file run.name=mine run.device=cuda 'loop.query_ratio=[0.05,0.1,0.2]' loop.save_final_model=true
```

Each round writes `oracle/round_XX_queries.json` with the people to annotate and stops. Correct their keypoints in CVAT or Label Studio, keep the annotation ids, save the result as `round_XX_answers.json` in the same folder and run the command again. `final_estimator.pth` holds the adapted estimator after the last round.

[docs/guide.md](docs/guide.md) lists the commands and settings and covers the annotation format, other joint layouts, training and tuning.

## Citation

```
@InProceedings{VATL4Pose_WACV24,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Active Transfer Learning for Efficient Video-Specific Human Pose Estimation},
  booktitle    = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year         = {2024}}

@InProceedings{VATL4Pose_MVA23,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation},
  booktitle    = {2023 18th International Conference on Machine Vision and Applications (MVA)},
  year         = {2023}}
```

## License

The source is under the MIT License, and `third_party/hrnet/` keeps its own MIT license. The weights are trained on PoseTrack21 and JRDB-Pose and are for noncommercial research only, under the terms of these datasets. The repository history and the tag `wacv2024` contain third-party code under its own licenses.

This work builds on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), [ALiPy](https://github.com/NUAA-AL/ALiPy), [DeepAL+](https://github.com/SineZHAN/deepALplus) and [VL4Pose](https://github.com/meghshukla/ActiveLearningForHumanPose).
