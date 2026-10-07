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

The first command downloads the weights of the release, checks each file by SHA256 and installs six files below `pretrained_models/`. The files are the estimators SimpleBaseline, FastPose and HRNet-W32 for PoseTrack21, FastPose for JRDB-Pose, and the whole-body autoencoder (Hybrid, zdim 4) of each dataset. The autoencoders are the `WholeBodyAE_zdim4_v1.pth` files. The autoencoder of this version standardises the features, ignores invisible joints and is retrained. The stored training features of the 2023 autoencoders hold one repeated row, and the released JRDB-Pose autoencoder returns a near-constant pose; the files of the paper do not load. `--from-file PATH` installs from a zip that you downloaded yourself. The second command checks the whole setup and exits with status 1 if a check fails.

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

`vatl4pose-atl` writes `outputs/<experiment>/<estimator>/<strategy>/<video>/result.json` for each video, with the strategy name from the second column of the table below, for example `THC+WPU_Coresetfilter`. The command skips a finished video and resumes an interrupted one. `Ctrl-C`, `SIGTERM` and `SIGHUP` stop it and release the lock of the video. After `SIGKILL`, an out-of-memory kill or a power loss, the lock file `<video>/.lock` stays and the next start stops with an error that names the file, its host and its pid. Check that no process with that host and pid is running, then delete the file by hand. To use several GPUs, start one process per GPU with `CUDA_VISIBLE_DEVICES=<n> ... run.shard=<n>/<number of GPUs>`. `vatl4pose-summarize` writes `summary.md`, `summary.json` and the learning curves to `outputs/<experiment>/summary/<estimator>/`. The strategies `thc`, `wpu`, `thc_wpu`, `thc_coreset`, `wpu_coreset` (the ablation of the paper) and `vl4pose` run the same way. `estimator=hrnet_w32` defines no embedding of its backbone, so it runs only with `random`, `hp`, `mpe`, `tpc`, `thc`, `wpu` and `thc_wpu`. With the default strategy and the other strategies, which cluster embeddings or read backbone features, the command stops with an error before round 0.

The table gives the paper values of the ALC of AP@0.6 in percent, the mean over the videos, and in its last row the round-0 AP@0.6 of the 2023 runs. The ALC is the area under the learning curve of AP@0.6 over the share of labeled people, and the labeled people count as correct. `summary.md` shows it in the column `ALC AP .6`.

| `strategy=` | name in `summary.md` | name in the paper | PoseTrack21, SimpleBaseline, 170 videos | JRDB-Pose, FastPose, 15 videos |
|---|---|---|---|---|
| `random` | Random | Random | 96.91 | 95.42 |
| `hp` | HP | LC | 95.74 | 92.67 |
| `mpe` | MPE | MPE | 96.11 | 95.76 |
| `tpc` | TPC | TPC | 96.40 | 93.76 |
| `kmeans` | _K-Meansfilter | k-means | 97.65 | 96.41 |
| `coreset` | _Coresetfilter | Core-Set | 98.12 | 95.60 |
| `thc_wpu_duw` | THC+WPU_Coresetfilter | THC+WPU+DUW | 98.21 | 96.52 |
| all | all | AP@0.6 before any annotation (round 0), 2023 runs, scale 0 to 1 | - | 0.561 |

The last row is the mean over the 15 JRDB-Pose test videos of FastPose with the released weights. The value of one video is `performances[0]["AP .6"]` in its `result.json`. The 2023 code evaluated JRDB-Pose with the OKS sigmas of COCO, and this version uses the OKS sigmas of JRDB-Pose. The JRDB-Pose sigmas raise the round-0 AP@0.6 of video 00 from 0.592 to 0.602 with `dataset.panorama=false`; the default `dataset.panorama=true` of JRDB-Pose gives 0.610 for this video (next paragraph).

The frames of JRDB-Pose are 360-degree panoramas, and this version places the joints that the file records on the other side of the seam next to their person and crops across the seam, which the 2023 code did not do. `dataset=jrdb_pose` sets `dataset.panorama=true`, and `dataset.panorama=false` reads the dataset as the 2023 code did.

The AP of this package is the AP of the paper, not the standard COCO keypoint AP. The 2023 evaluation gives a detection the area `(w - x) * (h - y)` of its box `[x, y, w, h]`, so a detection that no label matches and whose area lies outside the area range counts as ignored and not as a false positive. This version keeps that rule so that its numbers stay comparable with the paper. With the standard area `w * h`, the round-0 AP@0.6 of JRDB-Pose video 00 is 0.458 instead of 0.592 on the same predictions. Every AP of this package follows the rule, in `result.json`, in `vatl4pose-eval` and in the validation of `vatl4pose-train`, and `result.json` and the file of `vatl4pose-eval` name it in `"ap_protocol": "wacv2024-detection-area"`. An AP from another COCO keypoint evaluation is therefore not comparable. This package computes the ten values itself, because the summary of the stock `pycocotools` lists AP75 at the position of the key `AP .6`, which caused issue #40.

## Using your own data

Extract the frames of each video to image files first, and put the frames and one annotation file per video in this layout. `dataset.root` is the dataset directory, relative to `paths.data_root` unless it is an absolute path.

```
data/mine/images/clip1/000001.jpg
data/mine/annotations/clip1.json
```

`annotations/<video>.json` is a COCO keypoints file with track ids. The reader takes these fields and ignores the others.

- `images`: `id`, `file_name` (relative to `dataset.root`, for example `images/clip1/000001.jpg`, unique in the file), `width`, `height`.
- `annotations`: `id` (integer, unique in the file), `image_id`, `track_id` (all integers or all strings in a file), `bbox` as `[x, y, w, h]`, and optionally `keypoints` as a flat list of `x, y, visibility` for every joint. Leave the key out for a person without ground truth, because `[]` and `null` raise an error.

The reader orders the frames by the sorted `file_name`, so number the frames with zero padding (`000001.jpg`, not `1.jpg`). The boxes and the track ids come from your own detector and tracker, and an annotation with an empty box is skipped.

For a 360-degree video, whose right edge continues at its left edge, set `dataset.panorama=true`: the reader then moves the joints of a person on the seam next to the box of the person, and the crops continue across the seam.

With `oracle=coco_file` the keypoints of this file are not labels. The loop reads them only to report AP and OSPA. If any person has no visible joint, `result.json` holds `null` for these metrics and `vatl4pose-summarize` lists the video as "no ground truth". Selection, annotation rounds and retraining run as usual, while `oracle=simulated`, `vatl4pose-tune`, `vatl4pose-eval` and `vatl4pose-train` need the keypoints and stop with an error for such a video.

`dataset.keypoints` gives the joint layout. The value is the name of a preset (`coco17`, `posetrack21`, `jrdb_pose`) or a mapping with all seven fields of a keypoint spec. Joint indices count from 0. The joint order of a preset is the order of `joint_names` in `src/vatl4pose/keypoints/spec.py`. `coco17` is the COCO order. `posetrack21` has the same order except at joints 1 and 2, which are `head_bottom` and `head_top` instead of the eyes. `jrdb_pose` has its own order. A file in COCO order needs `coco17` or a custom spec.

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

`dataset.keypoints` also works for `dataset=posetrack21` and `dataset=jrdb_pose`, where it replaces the preset (for example to test other flip pairs or OKS sigmas). The annotation files must have the joint count of the spec. A cache belongs to one spec, so build it again with `vatl4pose-prepare` and another `paths.output_root`, and give each spec its own `run.name`.

The released estimators and autoencoders match the joint orders of `posetrack21` and `jrdb_pose`. Data in COCO order, or in another layout, needs an estimator trained for that layout. Write the ids of the training and the validation videos into two text files with one id per line, here `train_ids.txt` and `val_ids.txt`. These videos need `keypoints`.

```
vatl4pose-train dataset=coco_custom dataset.root=mine dataset.keypoints=coco17 \
  estimator=simplebaseline estimator.weights=null estimator.imagenet_init=true \
  train.target=estimator train.train_ids_file=train_ids.txt train.val_ids_file=val_ids.txt \
  run.name=mine_estimator run.device=cuda
```

`estimator.imagenet_init=true` starts from the torchvision ImageNet weights, which the command downloads on first use. The command writes `best.pth` (the epoch with the best validation AP), `last.pth`, `log.jsonl` and `summary.json` to `outputs/train/mine_estimator/`, and it refuses a directory that already holds a run. `logger=wandb` (install the extra with `uv sync --extra wandb`) also sends the loss, the learning rate, the ten AP values and the OSPA of every epoch to a W&B run named `<MMDD>_<dataset>_<run.name>`, and the same option sends every round of `vatl4pose-atl` and `vatl4pose-tune` to W&B. `vatl4pose-eval` refuses `logger=wandb`. The recipe of the paper is in `src/vatl4pose/configs/train/`. The autoencoder of WPU is trained with `train.target=ae`, and you can skip it if you use a strategy without WPU such as `strategy=thc_coreset`.

```
vatl4pose-train dataset=coco_custom dataset.root=mine dataset.keypoints=coco17 \
  train.target=ae train.train_ids_file=train_ids.txt train.val_ids_file=val_ids.txt \
  run.name=mine_ae run.device=cuda
```

Pass the trained files to `vatl4pose-atl` as absolute paths, for example `estimator.weights=/home/me/VATL4Pose-WACV2024/outputs/train/mine_estimator/best.pth` and `ae.checkpoint=/home/me/VATL4Pose-WACV2024/outputs/train/mine_ae/best.pth`. The options `estimator.weights` and `ae.checkpoint` resolve relative to `paths.weights_root`, so a path that starts with `outputs/` raises a `FileNotFoundError`.

The example below uses the `posetrack21` layout and the weights from `vatl4pose-download-weights` on two videos, `clip1` and `clip2`, and ends at 20 % labeled people. `loop.query_ratio` lists the share of labeled people after each round, and the default `[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]` ends with all people. `loop.save_final_model=true` makes the run write `final_estimator.pth` into the run directory of each video, next to `result.json`. The file holds the weights of the estimator adapted to that video.

```
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21

vatl4pose-atl dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  oracle=coco_file run.name=mine run.device=cuda 'loop.query_ratio=[0.05,0.1,0.2]' loop.save_final_model=true
```

The proposed strategy `thc_wpu_duw` has one setting that depends on the data, `lam`, the weight of the uncertainty term of DUW. The experiment configs set `lam=0.01` for PoseTrack21 and `lam=1000` for JRDB-Pose, and your own data uses `lam=0.01` unless you pass `lam=<value>`. `vatl4pose-tune` chooses `lam` for your data when you have annotated videos to tune on. For every value in `tune.lams` it runs the loop with `oracle=simulated` on the videos of `tune.ids_file`, and it writes the ALC of `tune.metric` (default `AP .95`) for each value to `outputs/<run.name>/tune/<estimator>/<strategy>/tune.md`. Tuning needs the keypoints of every person in its videos, because the simulated oracle answers with them and the objective is an AP, so the command stops with an error for a video without them. Without annotated videos, keep `lam=0.01`. Build the caches of the tuning videos with `vatl4pose-prepare`, and give the best value to `vatl4pose-atl` as `lam=<value>`.

```
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[val1,val2]' dataset.keypoints=posetrack21

vatl4pose-tune dataset=coco_custom dataset.root=mine dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  tune.ids_file=val_ids.txt 'tune.lams=[0.01,1,100]' run.name=mine_tune run.device=cuda
```

For the two paper datasets, the tuning videos are the `train_val` split of PoseTrack21 and the `val` split of JRDB-Pose. Their caches come from `vatl4pose-prepare experiment=wacv2024_posetrack21 dataset.split=train_val 'dataset.ids_file=${tune.ids_file}'` (and `dataset.split=val` with `experiment=wacv2024_jrdb`), and `vatl4pose-tune experiment=wacv2024_posetrack21 'tune.lams=[0.01,0.1,1]'` then tunes on them.

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

This work builds on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), [ALiPy](https://github.com/NUAA-AL/ALiPy), [DeepAL+](https://github.com/SineZHAN/deepALplus) and [VL4Pose](https://github.com/meghshukla/ActiveLearningForHumanPose), whose authors we thank.
