# VATL4Pose

VATL4Pose adapts a pretrained human pose estimator to one video with active transfer learning. In each round the method selects the people whom an annotator should label, using the temporal continuity of their heatmaps (THC), the unnaturalness of their whole-body pose (WPU) and a Core-Set selection with a dynamic uncertainty weight (DUW). The method then retrains the estimator on the new labels. This repository is the official implementation of [Active Transfer Learning for Efficient Video-Specific Human Pose Estimation](https://arxiv.org/abs/2311.05041) (WACV 2024, [project page](https://iminthemiddle.github.io/VATL4Pose-Page/)). The repository also runs the baselines of the paper and VL4Pose. The code of the paper version stays at the tag `wacv2024`.

## Installation

[uv](https://docs.astral.sh/uv/) installs Python 3.12 and the dependencies.

```
git clone https://github.com/ImIntheMiddle/VATL4Pose-WACV2024.git
cd VATL4Pose-WACV2024
uv sync --extra cu124
source .venv/bin/activate
```

The extra selects the torch build. `cu124` installs torch 2.6.0 for CUDA 12.4, `cu128` installs torch 2.7.1 for CUDA 12.8, and `cpu` installs torch 2.6.0 without CUDA. Read the "CUDA Version" in the header of `nvidia-smi`, which is the newest CUDA version your driver supports. Take `cu128` if it shows 12.8 or higher and `cu124` if it shows 12.4 to 12.7. If it shows less than 12.4, update the driver.

Without `source .venv/bin/activate`, put `uv run` in front of every command below. The commands read `paths.data_root` (default `data`), `paths.weights_root` (default `pretrained_models`) and `paths.output_root` (default `outputs`), relative to the directory you run them in. Set another place with, for example, `paths.data_root=/mnt/datasets`.

```
vatl4pose-doctor paths.weights_root=pretrained_models
```

The first two rows of the output show the torch build and whether CUDA is visible. The weight rows fail until you have installed the weights (see Reproducing the paper).

## Data preparation

Download PoseTrack21 and JRDB-Pose and put them under `data/`.

- PoseTrack21: https://github.com/anDoer/PoseTrack21
- JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose

```
data/PoseTrack21/posetrack_data/{train,val}/   one JSON file per video
data/PoseTrack21/images/{train,val}/           one folder per video
data/jrdb-pose/jrdb2022/labels/                labels_2d_pose_stitched_coco/ and labels_2d_stitched/
data/jrdb-pose/images/image_stitched/          one folder per scene
```

`vatl4pose-prepare-annotations` turns the official releases into one JSON file per video, `<dataset>/activelearning/<split>/<video>_*.json`, which is the file the other commands read. The file of a PoseTrack21 validation video keeps the labeled frames within the 33 frames around the middle frame. The file of a JRDB-Pose scene keeps the first 149 frames.

```
vatl4pose-prepare-annotations experiment=wacv2024_posetrack21
vatl4pose-prepare-annotations experiment=wacv2024_jrdb
```

These two commands prepare the 170 PoseTrack21 validation videos and the 15 JRDB-Pose test videos of the paper experiments. The splits for training and tuning follow with `vatl4pose-prepare-annotations dataset=posetrack21 dataset.split=train` (or `train_val`) and `dataset=jrdb_pose dataset.split=train` (or `val`). The id lists packaged with vatl4pose cut the 593 training videos of PoseTrack21 into 579 videos for `train` and 14 for `train_val`.

`vatl4pose-prepare` then builds one HDF5 cache per video below `outputs/cache/`, which `vatl4pose-atl` reads. The command keeps a cache that matches its video.

```
vatl4pose-prepare experiment=wacv2024_posetrack21
vatl4pose-prepare experiment=wacv2024_jrdb
```

## Reproducing the paper

```
vatl4pose-download-weights --weights-root pretrained_models
vatl4pose-doctor paths.weights_root=pretrained_models paths.data_root=data
```

The first command downloads the weights of the release, checks each file by SHA256 and installs six files below `pretrained_models/`. They are the estimators SimpleBaseline, FastPose and HRNet-W32 for PoseTrack21, FastPose for JRDB-Pose, and the whole-body autoencoder (Hybrid, zdim 4) of each dataset. `--from-file PATH` installs from a zip that you downloaded yourself. The second command checks the whole setup and exits with status 1 if a check fails.

The paper evaluates SimpleBaseline on PoseTrack21 (Table 2) and FastPose on JRDB-Pose (Table 4). `strategy=thc_wpu_duw` is the proposed method and the default.

```
for s in random hp mpe tpc kmeans coreset thc_wpu_duw; do
  vatl4pose-atl experiment=wacv2024_posetrack21 strategy=$s run.device=cuda
done
vatl4pose-summarize experiment=wacv2024_posetrack21

for s in random hp mpe tpc kmeans coreset thc_wpu_duw; do
  vatl4pose-atl experiment=wacv2024_jrdb strategy=$s run.device=cuda
done
vatl4pose-summarize experiment=wacv2024_jrdb
```

`vatl4pose-atl` writes `outputs/<experiment>/<estimator>/<strategy>/<video>/result.json` for each video, with the strategy name from the second column of the table below, for example `THC+WPU_Coresetfilter`. The command skips a finished video and resumes an interrupted one. To use several GPUs, start one process per GPU with `CUDA_VISIBLE_DEVICES=<n> ... run.shard=<n>/<number of GPUs>`. `vatl4pose-summarize` writes `summary.md`, `summary.json` and the learning curves to `outputs/<experiment>/summary/<estimator>/`. The strategies `thc`, `wpu`, `thc_wpu`, `thc_coreset`, `wpu_coreset` (the ablation of the paper) and `vl4pose` run the same way.

The table gives the paper values of the ALC of AP@0.6 in percent, the mean over the videos. The ALC is the area under the learning curve of AP@0.6 over the share of labeled people, and the labeled people count as correct. `summary.md` shows it in the column `ALC AP .6`.

| `strategy=` | name in `summary.md` | name in the paper | PoseTrack21, SimpleBaseline, 170 videos | JRDB-Pose, FastPose, 15 videos |
|---|---|---|---|---|
| `random` | Random | Random | 96.91 | 95.42 |
| `hp` | HP | LC | 95.74 | 92.67 |
| `mpe` | MPE | MPE | 96.11 | 95.76 |
| `tpc` | TPC | TPC | 96.40 | 93.76 |
| `kmeans` | _K-Meansfilter | k-means | 97.65 | 96.41 |
| `coreset` | _Coresetfilter | Core-Set | 98.12 | 95.60 |
| `thc_wpu_duw` | THC+WPU_Coresetfilter | THC+WPU+DUW | 98.21 | 96.52 |

Before any annotation (round 0), FastPose with the JRDB-Pose weights reaches a mean AP@0.6 of 0.561 over the 15 test videos in the 2023 runs. The value of one video is `performances[0]["AP .6"]` in its `result.json`. The 2023 code evaluated JRDB-Pose with the OKS sigmas of COCO, and this version uses the OKS sigmas of JRDB-Pose. The JRDB-Pose sigmas raise the round-0 AP@0.6 of video 00 from 0.592 to 0.602.

## Using your own data

Put your videos in this layout. `dataset.root` is the dataset directory, relative to `paths.data_root` unless it is an absolute path.

```
data/mine/images/<video>/<frame>.jpg
data/mine/annotations/<video>.json
```

`annotations/<video>.json` is a COCO keypoints file with track ids. The reader takes these fields and ignores the others.

- `images`: `id`, `file_name` (relative to `dataset.root`, unique in the file), `width`, `height`.
- `annotations`: `id` (integer, unique in the file), `image_id`, `track_id` (all integers or all strings in a file), `bbox` as `[x, y, w, h]`, and optionally `keypoints` as a flat list of `x, y, visibility` for every joint.

The boxes and the track ids come from your own detector and tracker, and boxes alone are enough. An annotation with an empty box is skipped. An annotation without `keypoints`, or without a visible joint, is a person without ground truth.

With `oracle=coco_file` the keypoints of this file are not used as labels. They only give the AP and OSPA that the loop reports. If every person has a visible joint, the loop reports both as for the paper datasets. If one person or more have none, the loop computes no AP and no OSPA. It logs one warning at each start of a run, `result.json` holds `null` for `performances`, `performances_ann`, `ospa` and `ospa_ann` in every round, and `vatl4pose-summarize` lists the video as "no ground truth" and leaves it out of the means. The selection, the annotation rounds and the retraining do not read the ground truth and run as usual. `oracle=simulated`, `vatl4pose-tune`, `vatl4pose-eval` and `vatl4pose-train` answer or score with the keypoints, so they stop with an error for a video that has a person without them.

`dataset.keypoints` gives the joint layout. The value is the name of a preset (`coco17`, `posetrack21`, `jrdb_pose`) or a mapping with all seven fields of a keypoint spec. Joint indices count from 0.

| field | content |
|---|---|
| `name` | name of the layout |
| `joint_names` | one unique name per joint, in the order of `keypoints` |
| `flip_pairs` | pairs of joints that swap places in a horizontal flip |
| `skeleton` | pairs of joints that a limb connects, used for drawing and for the annotation tool |
| `oks_sigmas` | OKS sigma of each joint divided by 10 (0.026 for the COCO nose) |
| `redundant_joints` | joints that are never annotated, left out of the WPU error |
| `angle_triplets` | triplets `(a, b, c)` of joints; the feature holds the angle at `b` between `a-b` and `b-c` |

```
'dataset.keypoints={name:three,joint_names:[a,b,c],flip_pairs:[[0,1]],skeleton:[[0,2]],oks_sigmas:[0.05,0.05,0.05],redundant_joints:[],angle_triplets:[[0,2,1]]}'
```

The released estimators and autoencoders fit the layouts `posetrack21` and `jrdb_pose`. For another layout, train an estimator with `vatl4pose-train train.target=estimator` (see `vatl4pose-train --help` and `src/vatl4pose/configs/train/`). Train an autoencoder with `train.target=ae` as well, unless you use a strategy without WPU such as `strategy=thc_coreset`. The result of `vatl4pose-train` is `outputs/train/<run.name>/best.pth`, which you give as `estimator.weights` or `ae.checkpoint`.

The example below uses the PoseTrack21 layout and weights on two videos, `clip1` and `clip2`, and ends at 20 % labeled people. `loop.query_ratio` lists the share of labeled people after each round, and the default `[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]` ends with all people.

```
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21

vatl4pose-atl dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4.pth \
  oracle=coco_file run.name=mine run.device=cuda 'loop.query_ratio=[0.05,0.1,0.2]'
```

With `oracle=coco_file` a person annotates the queries of every round.

1. The run writes `outputs/<run.name>/<estimator>/<strategy>/<video>/oracle/round_XX_queries.json` for each video. The file is a COCO keypoints file with the selected people. Each person carries the model's prediction as a starting point. The command goes on with the next video, prints a `Pending` line for each video and exits with status 0.
2. Import the queries into CVAT or Label Studio as COCO Keypoints, correct the keypoints, and export the result as COCO Keypoints.
3. Save the export as `round_XX_answers.json` in the same `oracle/` folder.
4. Run the same command again. The run reads the answers, retrains the estimator and writes the queries of the next round. After the last round it writes `result.json`.

The loop reads each answer by the annotation `id` of the queries file, and `id` and `keypoints` are the only fields it uses. The answers have to contain every queried `id` once and no other `id`, and `keypoints` holds `x, y, visibility` for every joint with a visibility of 0, 1 or 2. The annotation tool has to keep the `id` values in its export. CVAT and Label Studio have not been checked for this. A rerun with other settings than the first run stops with an error that names the changed keys. Give new settings a new `run.name`.

## Citation

If you use this code, please cite our work.

```
@InProceedings{VATL4Pose_WACV24,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Active Transfer Learning for Efficient Video-Specific Human Pose Estimation},
  booktitle    = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year         = {2024}}
```

The earlier paper on the uncertainty criteria is MVA 2023 ([IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10215565)).

```
@InProceedings{VATL4Pose_MVA23,
  author       = {Taketsugu, Hiromu and Ukita, Norimichi},
  title        = {Uncertainty Criteria in Active Transfer Learning for Efficient Video-Specific Human Pose Estimation},
  booktitle    = {2023 18th International Conference on Machine Vision and Applications (MVA)},
  year         = {2023}}
```

## License

This repository is released under the MIT License (`LICENSE`). `third_party/hrnet/` holds the HRNet-W32 network definition of [deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) under its MIT license (`third_party/hrnet/LICENSE`), and `third_party/hrnet/UPSTREAM.md` lists the source commit and every local change. VL4Pose and AlphaPose are reimplemented here and their code is not included.

This work builds on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), [ALiPy](https://github.com/NUAA-AL/ALiPy), [DeepAL+](https://github.com/SineZHAN/deepALplus) and [VL4Pose](https://github.com/meghshukla/ActiveLearningForHumanPose). We thank their authors.
