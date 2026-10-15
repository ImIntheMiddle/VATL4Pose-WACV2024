# Guide

## Commands

| command | what it does |
|---|---|
| `vatl4pose-download-weights` | installs the released weights (`--from-file PATH` installs a zip you downloaded) |
| `vatl4pose-doctor` | checks the environment, the weights and the data |
| `vatl4pose-prepare-annotations` | converts PoseTrack21 or JRDB-Pose into one annotation file per video |
| `vatl4pose-prepare` | builds the cache of each video in `outputs/cache/` |
| `vatl4pose-atl` | runs active transfer learning and writes `outputs/<run.name>/<estimator>/<strategy>/<video>/result.json` |
| `vatl4pose-summarize` | writes `summary.md` and the learning curves |
| `vatl4pose-train` | trains an estimator (`train.target=estimator`) or the WPU autoencoder (`train.target=ae`) |
| `vatl4pose-tune` | chooses `lam` of the proposed method on annotated videos |
| `vatl4pose-eval` | evaluates an estimator |
| `vatl4pose-explain` | draws the people that each round selected and the reasons, from the records of `loop.save_round_details=true` |
| `vatl4pose annotate` | annotates a video of your own in the browser (see [annotation/README.md](../annotation/README.md)) |
| `vatl4pose run / status / resume` | runs a workflow of several commands (see the end of this guide) |

Settings are Hydra overrides (`key=value`). The main ones are below.

| setting | values |
|---|---|
| `strategy` | `thc_wpu_duw` (proposed, default), `random`, `hp`, `mpe`, `tpc`, `kmeans`, `coreset`, `thc`, `wpu`, `thc_wpu`, `thc_coreset`, `wpu_coreset`, `vl4pose`; the ablation of Table 3: `thc_wpu_duw_fixed`, `thc_wpu_duw_increase`, `thc_wpu_duw_decrease` |
| `estimator` | `simplebaseline`, `fastpose`, `hrnet_w32` (`hrnet_w32` has no embedding and works only with `random`, `hp`, `mpe`, `tpc`, `thc`, `wpu`, `thc_wpu`) |
| `oracle` | `simulated` (answers with the ground truth), `coco_file` (a person answers) |
| `paths.data_root`, `paths.weights_root`, `paths.output_root` | `data`, `pretrained_models`, `outputs` |
| `loop.query_ratio` | share of labeled people after each round, default `[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]` |
| `loop.save_final_model` | `true` writes `final_estimator.pth` next to `result.json` |
| `loop.save_estimator` | `true` writes `estimator.pth` after every round |
| `loop.stopping` | `true` judges the stopping criteria of the paper in every round (see below) |
| `loop.save_round_details` | `true` writes `details/round_XX.npz` in every round for `vatl4pose-explain` |
| `retrain.continual` | `true` (default) keeps the model, the optimizer and the schedule from round to round; `false` starts every round from `estimator.weights` |
| `retrain.optimizer` | `adamw` (default), `adam`, `sgd` |
| `retrain.finish_acc` | OKS θ from which a labeled person counts as solved: people labeled in earlier rounds are retrained only while their OKS is at most θ + 0.05 (default 1.0, all of them) |
| `dataset.panorama` | `true` for 360-degree frames whose right edge continues at the left edge (JRDB-Pose) |
| `logger` | `wandb` logs to W&B (`uv sync --extra wandb`) |

`estimator.weights` and `ae.checkpoint` are relative to `paths.weights_root`. Give your own files as absolute paths. `estimator.weights=null estimator.imagenet_init=true` starts the loop from the ImageNet backbone instead of a pose estimator, the run without transfer. To use several GPUs, start one process per GPU with `CUDA_VISIBLE_DEVICES=<n> ... run.shard=<n>/<number of GPUs>`. A run resumes where it stopped. If a process was killed, delete the `<video>/.lock` file that the error names.

The AP follows the evaluation of the paper (`"ap_protocol": "wacv2024-detection-area"` in `result.json`) and differs from the standard COCO keypoint AP.

## Output files

The run directory of a video, `outputs/<run.name>/<estimator>/<strategy>/<video>/`, holds these files.

| file | content |
|---|---|
| `result.json` | the share of labeled people, AP, OSPA and the selected people of every round |
| `log.jsonl` | one line per round, written while the run goes on |
| `predictions.json` | after the last round: the keypoints of every person under the final estimator, as a COCO results list with the annotation `id` and `track_id` of each person |
| `predictions_ann.json` | the same, with the labels of the annotated people in place of the estimate |
| `final_estimator.pth`, `estimator.pth` | the estimator after the last round, and after every round (`loop.save_final_model`, `loop.save_estimator`) |
| `oracle/` | the queries and answers of `oracle=coco_file` |
| `details/` | the records of `loop.save_round_details` |

For JRDB-Pose the x of a joint stays next to the box of its person, as in the evaluation, so a joint across the seam can lie below 0 or beyond the image width.

## Tables 3 and 5 of the paper

The ablation of Table 3 changes one part of the proposed method. `thc_wpu_duw_fixed` drops the dynamic weight of DUW and picks the person with the largest distance plus `lam` times the uncertainty. `thc_wpu_duw_increase` and `thc_wpu_duw_decrease` weigh THC and WPU by the share r of labeled people, THC by r and WPU by 1 - r, or the reverse.

```bash
for s in thc_wpu_duw_fixed thc_wpu_duw_increase thc_wpu_duw_decrease; do
  vatl4pose-atl experiment=wacv2024_posetrack21 strategy=$s run.device=cuda
done
vatl4pose-summarize experiment=wacv2024_posetrack21
```

Table 5 compares stopping criteria with a target OKS θ. With `loop.stopping=true` every round records three judgments under the estimator before it learns from the round: SC_Min, whether the mean OKS of the people queried in the round reaches θ; SC_All, whether every person annotated so far reaches θ; and the actual stop, whether every person of the video reaches θ against the ground truth. The loop does not stop by itself. `vatl4pose-summarize` reports the mean share of labeled people at the stop of each criterion, and for SC_Min and SC_All the AP at OKS θ there. The preset `wacv2024_posetrack21_table5` raises the share in steps of 10 % as the runs of the paper did, and `retrain.finish_acc` sets θ.

```bash
for theta in 0.5 0.6 0.7 0.8; do
  vatl4pose-atl experiment=wacv2024_posetrack21_table5 retrain.finish_acc=$theta run.device=cuda
  vatl4pose-summarize experiment=wacv2024_posetrack21_table5 retrain.finish_acc=$theta
done
```

## Pictures of each round

With `loop.save_round_details=true`, `vatl4pose-atl` also writes `details/round_XX.npz` in every round. The records only read the results, so the selected people, the training and `result.json` stay the same. `vatl4pose-explain` draws them into `explain/` of the run directory.

```bash
uv sync --extra explain        # the H.264 encoder of the videos
vatl4pose-explain outputs/mine/simplebaseline/THC+WPU_Coresetfilter/clip1
```

- `index.html` opens from the file system. It switches between rounds and numbers the selected people; a click on a person shows the THC, WPU and DUW figures of that pick.
- `round_XX.mp4` and `compare_00_XX.mp4` show the predictions of a round, alone and beside round 0. They, the page and the pictures in `picks/` leave out the joints whose heatmap peak is below 0.3, mostly joints that the labels mark as not visible, whose predictions jump between frames.
- `picks/round_XX_rank_YY.png` explains one selected person in one picture.

`--rounds 0,2` selects rounds, `--image-root DIR` replaces the location of the frames, `--max-picks N` limits the pictures per round (40 by default), and `--no-video` skips the videos.

## Your own videos

The layout is the one in the [README](../README.md#your-own-videos): `images/<video>/` for the frames and `annotations/<video>.json` for a COCO keypoints file, both below `dataset.root`. The JSON file holds two lists.

- `images`: `id`, `file_name` (relative to `dataset.root`, for example `images/clip1/000001.jpg`), `width`, `height`. The frames are ordered by `file_name`, so pad the numbers with zeros.
- `annotations`: `id` (unique integer), `image_id`, `track_id`, `bbox` as `[x, y, w, h]`, and optionally `keypoints` as `x, y, visibility` for every joint. Leave `keypoints` out for a person without ground truth.

The boxes and track ids come from your detector and tracker. For a 360-degree video set `dataset.panorama=true` and let the box of a person on the seam extend past the image edge. Without ground truth, AP and OSPA are `null`, and `oracle=simulated`, `vatl4pose-tune`, `vatl4pose-eval` and `vatl4pose-train` cannot use the video.

## Annotating the queries

With `oracle=coco_file`, each round writes `oracle/round_XX_queries.json` in the run directory of each video and stops. The file holds the selected people with the predicted keypoints.

1. Correct the keypoints in CVAT or Label Studio.
2. Save the result as `round_XX_answers.json` in the same folder. It must contain every queried annotation `id` once, with `keypoints` as `x, y, visibility` (0, 1 or 2) for every joint. Label Studio's COCO export renumbers the ids, so export JSON and map the answers back by the query id.
3. Run the same command again. It retrains the estimator and writes the next queries.

Changed settings need a new `run.name`.

## Joint layouts

`dataset.keypoints` is a preset (`coco17`, `posetrack21`, `jrdb_pose`) or a mapping with these fields. The released weights use `posetrack21` and `jrdb_pose`.

| field | content |
|---|---|
| `name` | name of the layout |
| `joint_names` | one name per joint, in the order of `keypoints` |
| `flip_pairs` | joint pairs that swap in a horizontal flip |
| `skeleton` | joint pairs connected by a limb |
| `oks_sigmas` | OKS sigma of each joint |
| `redundant_joints` | joints that are never annotated |
| `angle_triplets` | joint triplets `(a, b, c)` whose angle at `b` enters the WPU feature |

```
'dataset.keypoints={name:three,joint_names:[a,b,c],flip_pairs:[[0,1]],skeleton:[[0,2]],oks_sigmas:[0.05,0.05,0.05],redundant_joints:[],angle_triplets:[[0,2,1]]}'
```

## Training an estimator and the autoencoder

Another layout needs its own estimator, and WPU needs its own autoencoder. List the training and validation video ids in two text files, one id per line. These videos need `keypoints`.

```bash
vatl4pose-train dataset=coco_custom dataset.root=mine dataset.keypoints=coco17 \
  estimator=simplebaseline estimator.weights=null estimator.imagenet_init=true \
  train.target=estimator train.train_ids_file=train_ids.txt train.val_ids_file=val_ids.txt \
  run.name=mine_estimator run.device=cuda

vatl4pose-train dataset=coco_custom dataset.root=mine dataset.keypoints=coco17 \
  train.target=ae train.train_ids_file=train_ids.txt train.val_ids_file=val_ids.txt \
  run.name=mine_ae run.device=cuda
```

Each command writes `best.pth` to `outputs/train/<run.name>/`. A strategy without WPU, such as `thc_coreset`, needs no autoencoder.

## Tuning `lam`

`lam` weights the uncertainty in DUW. Your own data uses `lam=0.01` unless you pass another value. With annotated videos, `vatl4pose-tune` runs the loop for each value and writes the result to `tune.md`.

```bash
vatl4pose-tune dataset=coco_custom dataset.root=mine dataset.keypoints=posetrack21 \
  estimator=simplebaseline estimator.weights=posetrack21/simplebaseline_256x192.pth \
  ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth \
  tune.ids_file=val_ids.txt 'tune.lams=[0.01,1,100]' run.name=mine_tune run.device=cuda
```

## Workflows

`vatl4pose run` runs the jobs of a YAML file in order of their dependencies, and `vatl4pose resume` continues a stopped workflow. With `--gpu`, the runner waits until the GPU shows no compute process in two checks 180 seconds apart.

```yaml
schema: vatl4pose.workflow/1
run_dir: outputs/my_video
jobs:
  - id: prepare
    argv: [vatl4pose-prepare, dataset=coco_custom, dataset.root=mine, "dataset.ids=[clip1]", dataset.keypoints=posetrack21]
  - id: atl
    depends_on: [prepare]
    gpu: true
    argv:
      - vatl4pose-atl
      - dataset=coco_custom
      - dataset.root=mine
      - "dataset.ids=[clip1]"
      - dataset.keypoints=posetrack21
      - estimator.weights=posetrack21/simplebaseline_256x192.pth
      - ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth
      - oracle=coco_file
      - run.device=cuda
```

```bash
vatl4pose run --config workflow.yaml --gpu 0
vatl4pose status outputs/my_video
vatl4pose resume outputs/my_video
```
