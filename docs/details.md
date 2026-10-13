# Details

## Installation

The extra selects the torch build. `cu124` installs torch 2.6.0 for CUDA 12.4, `cu128` installs torch 2.7.1 for CUDA 12.8, and `cpu` installs torch 2.6.0 without CUDA. Read the "CUDA Version" in the header of `nvidia-smi`, which is the newest CUDA version your driver supports. Take `cu128` if it shows 12.8 or higher and `cu124` if it shows 12.4 to 12.7. If it shows less than 12.4, update the driver.

Without `source .venv/bin/activate`, put `uv run` in front of every command. The commands read `paths.data_root` (default `data`), `paths.weights_root` (default `pretrained_models`) and `paths.output_root` (default `outputs`), relative to the directory you run them in. Set another place with, for example, `paths.data_root=/mnt/datasets`.

## Data

`vatl4pose-prepare-annotations` turns the official releases into one JSON file per video, `<dataset>/activelearning/<split>/<video>_*.json`, which is the file the other commands read. The file of a PoseTrack21 validation video keeps the labeled frames within the 33 frames around the middle frame. The file of a JRDB-Pose scene keeps the first 149 frames.

These two commands prepare the 170 PoseTrack21 validation videos and the 15 JRDB-Pose test videos of the paper experiments. The splits for training and tuning follow with `vatl4pose-prepare-annotations dataset=posetrack21 dataset.split=train` (or `train_val`) and `dataset=jrdb_pose dataset.split=train` (or `val`). The id lists packaged with vatl4pose cut the 593 training videos of PoseTrack21 into 579 videos for `train` and 14 for `train_val`.

`vatl4pose-prepare` then builds one HDF5 cache per video below `outputs/cache/`, which `vatl4pose-atl` reads. The command keeps a cache that matches its video.

## Weights

The weights are the estimators SimpleBaseline, FastPose and HRNet-W32 for PoseTrack21, FastPose for JRDB-Pose, and the whole-body autoencoder (Hybrid, zdim 4) of each dataset. The autoencoders are the `WholeBodyAE_zdim4_v1.pth` files. The autoencoder of this version standardises the features, ignores invisible joints and is retrained. The stored training features of the 2023 autoencoders hold one repeated row, and the released JRDB-Pose autoencoder returns a near-constant pose; the files of the paper do not load. `vatl4pose-download-weights --from-file PATH` installs from a zip that you downloaded yourself.

## Running the experiments

`vatl4pose-atl` writes `outputs/<experiment>/<estimator>/<strategy>/<video>/result.json` for each video, with the strategy name of `summary.md`, for example `THC+WPU_Coresetfilter`. The command skips a finished video and resumes an interrupted one. `Ctrl-C`, `SIGTERM` and `SIGHUP` stop it and release the lock of the video. After `SIGKILL`, an out-of-memory kill or a power loss, the lock file `<video>/.lock` stays and the next start stops with an error that names the file, its host and its pid. Check that no process with that host and pid is running, then delete the file by hand. To use several GPUs, start one process per GPU with `CUDA_VISIBLE_DEVICES=<n> ... run.shard=<n>/<number of GPUs>`. `vatl4pose-summarize` writes `summary.md`, `summary.json` and the learning curves to `outputs/<experiment>/summary/<estimator>/`. The strategies `thc`, `wpu`, `thc_wpu`, `thc_coreset`, `wpu_coreset` (the ablation of the paper) and `vl4pose` run the same way. `estimator=hrnet_w32` defines no embedding of its backbone, so it runs only with `random`, `hp`, `mpe`, `tpc`, `thc`, `wpu` and `thc_wpu`. With the default strategy and the other strategies, which cluster embeddings or read backbone features, the command stops with an error before round 0.

The ALC is the area under the learning curve of AP@0.6 over the share of labeled people, and the labeled people count as correct. `summary.md` shows it in the column `ALC AP .6`.

## Evaluation

The AP of this package is the AP of the paper, not the standard COCO keypoint AP. The 2023 evaluation gives a detection the area `(w - x) * (h - y)` of its box `[x, y, w, h]`, so a detection that no label matches and whose area lies outside the area range counts as ignored and not as a false positive. This version keeps that rule so that its numbers stay comparable with the paper. With the standard area `w * h`, the round-0 AP@0.6 of JRDB-Pose video 00 is 0.458 instead of 0.592 on the same predictions. Every AP of this package follows the rule, in `result.json`, in `vatl4pose-eval` and in the validation of `vatl4pose-train`, and `result.json` and the file of `vatl4pose-eval` name it in `"ap_protocol": "wacv2024-detection-area"`. An AP from another COCO keypoint evaluation is therefore not comparable. This package computes the ten values itself, because the summary of the stock `pycocotools` lists AP75 at the position of the key `AP .6`, which caused issue #40.

The 2023 code evaluated JRDB-Pose with the OKS sigmas of COCO, and this version uses the OKS sigmas of JRDB-Pose.

The frames of JRDB-Pose are 360-degree panoramas, and this version places the joints that the file records on the other side of the seam next to their person and crops across the seam, which the 2023 code did not do. `dataset=jrdb_pose` sets `dataset.panorama=true`, and `dataset.panorama=false` reads the dataset as the 2023 code did.
