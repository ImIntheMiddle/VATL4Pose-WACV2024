# Your own videos

## Annotation files

Extract the frames of each video to image files first, and put the frames and one annotation file per video in this layout. `dataset.root` is the dataset directory, relative to `paths.data_root` unless it is an absolute path.

```
data/mine/images/clip1/000001.jpg
data/mine/annotations/clip1.json
```

`annotations/<video>.json` is a COCO keypoints file with track ids. The reader takes these fields and ignores the others.

- `images`: `id`, `file_name` (relative to `dataset.root`, for example `images/clip1/000001.jpg`, unique in the file), `width`, `height`.
- `annotations`: `id` (integer, unique in the file), `image_id`, `track_id` (all integers or all strings in a file), `bbox` as `[x, y, w, h]`, and optionally `keypoints` as a flat list of `x, y, visibility` for every joint. Leave the key out for a person without ground truth, because `[]` and `null` raise an error.

The reader orders the frames by the sorted `file_name`, so number the frames with zero padding (`000001.jpg`, not `1.jpg`). The boxes and the track ids come from your own detector and tracker, and an annotation with an empty box is skipped.

For a 360-degree video, whose right edge continues at its left edge, set `dataset.panorama=true`: the reader then moves the joints of a person on the seam next to the box of the person, and the crops continue across the seam. Write the `bbox` of a person on the seam so that it extends past the edge (`x + w > W`, or `x < 0`) and is not cut at the edge; the reader keeps such a box as it is and clips it only at the top and the bottom.

With `oracle=coco_file` the keypoints of this file are not labels. The loop reads them only to report AP and OSPA. If any person has no visible joint, `result.json` holds `null` for these metrics and `vatl4pose-summarize` lists the video as "no ground truth". Selection, annotation rounds and retraining run as usual, while `oracle=simulated`, `vatl4pose-tune`, `vatl4pose-eval` and `vatl4pose-train` need the keypoints and stop with an error for such a video.

## Joint layouts

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

## Training an estimator and an autoencoder

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

With `estimator=hrnet_w32 estimator.weights=null`, the network uses PyTorch's default layer initialisation. The upstream HRNet normal(0.001) initialisation is not applied. Use the released HRNet checkpoint for the pretrained reference network.

Pass the trained files to `vatl4pose-atl` as absolute paths, for example `estimator.weights=/home/me/VATL4Pose-WACV2024/outputs/train/mine_estimator/best.pth` and `ae.checkpoint=/home/me/VATL4Pose-WACV2024/outputs/train/mine_ae/best.pth`. The options `estimator.weights` and `ae.checkpoint` resolve relative to `paths.weights_root`, so a path that starts with `outputs/` raises a `FileNotFoundError`.

## Running the loop

The example below uses the `posetrack21` layout and the weights from `vatl4pose-download-weights` on two videos, `clip1` and `clip2`, and ends at 20 % labeled people. `loop.query_ratio` lists the share of labeled people after each round, and the default `[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]` ends with all people. `loop.save_final_model=true` makes the run write `final_estimator.pth` into the run directory of each video, next to `result.json`. The file holds the weights of the estimator adapted to that video.

```
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21

vatl4pose-atl dataset=coco_custom dataset.root=mine 'dataset.ids=[clip1,clip2]' dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  oracle=coco_file run.name=mine run.device=cuda 'loop.query_ratio=[0.05,0.1,0.2]' loop.save_final_model=true
```

## Tuning `lam`

The proposed strategy `thc_wpu_duw` has one setting that depends on the data, `lam`, the weight of the uncertainty term of DUW. The experiment configs set `lam=0.01` for PoseTrack21 and `lam=1000` for JRDB-Pose, and your own data uses `lam=0.01` unless you pass `lam=<value>`. `vatl4pose-tune` chooses `lam` for your data when you have annotated videos to tune on. For every value in `tune.lams` it runs the loop with `oracle=simulated` on the videos of `tune.ids_file`, and it writes the ALC of `tune.metric` (default `AP .95`) for each value to `outputs/<run.name>/tune/<estimator>/<strategy>/tune.md`. Tuning needs the keypoints of every person in its videos, because the simulated oracle answers with them and the objective is an AP, so the command stops with an error for a video without them. Without annotated videos, keep `lam=0.01`. Build the caches of the tuning videos with `vatl4pose-prepare`, and give the best value to `vatl4pose-atl` as `lam=<value>`.

```
vatl4pose-prepare dataset=coco_custom dataset.root=mine 'dataset.ids=[val1,val2]' dataset.keypoints=posetrack21

vatl4pose-tune dataset=coco_custom dataset.root=mine dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  tune.ids_file=val_ids.txt 'tune.lams=[0.01,1,100]' run.name=mine_tune run.device=cuda
```

For the two paper datasets, the tuning videos are the `train_val` split of PoseTrack21 and the `val` split of JRDB-Pose. Their caches come from `vatl4pose-prepare experiment=wacv2024_posetrack21 dataset.split=train_val 'dataset.ids_file=${tune.ids_file}'` (and `dataset.split=val` with `experiment=wacv2024_jrdb`), and `vatl4pose-tune experiment=wacv2024_posetrack21 'tune.lams=[0.01,0.1,1]'` then tunes on them.

## Annotating the queries

With `oracle=coco_file` a person annotates the queries of every round.

1. The run writes `outputs/<run.name>/<estimator>/<strategy>/<video>/oracle/round_XX_queries.json` for each video. The file is a COCO keypoints file with the selected people. Each person carries the model's prediction as a starting point. The command goes on with the next video, prints a `Pending` line for each video and exits with status 0.
2. Import the queries into CVAT or Label Studio as COCO Keypoints, correct the keypoints, and export the result as COCO Keypoints.
3. Save the export as `round_XX_answers.json` in the same `oracle/` folder.
4. Run the same command again. The run reads the answers, retrains the estimator and writes the queries of the next round. After the last round it writes `result.json`.

The loop reads each answer by the annotation `id` of the queries file, and `id` and `keypoints` are the only fields it uses. The answers have to contain every queried `id` once and no other `id`, and `keypoints` holds `x, y, visibility` for every joint with a visibility of 0, 1 or 2. The annotation tool has to keep the `id` values in its export, or an adapter must restore them. Label Studio's COCO export renumbers the annotation ids: keep the query id in each task's data, export JSON and convert the answers using that query id. CVAT's id preservation has not been checked. A rerun with other settings than the first run stops with an error that names the changed keys. Give new settings a new `run.name`.
