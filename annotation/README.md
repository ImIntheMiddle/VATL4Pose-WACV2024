# Annotation tools for VATL4Pose

These tools annotate your own video and adapt the pretrained estimator of VATL4Pose to it in one local browser session. They live in the `annotation/` folder of the repository, next to the `vatl4pose` package, which holds the active learning loop and the `vatl4pose annotate` command.

A session detects the people with a torchvision Faster R-CNN and tracks them with [SAMIDARE](https://github.com/ZabuZabuZabu/SAMIDARE) ([CVsports 2026](https://arxiv.org/abs/2604.22162)), a tracking-by-segmentation method for crowded scenes that runs on the video predictor of [SAM 2](https://github.com/facebookresearch/sam2). The loop of VATL4Pose then selects the people to annotate, and a browser page shows the Label Studio editor with their predicted keypoints. You correct the points and press Train. The loop retrains the estimator on your corrections, and the page shows the keypoints that the retrained estimator predicts for everyone. You then start the next round or finish.

The tools support the joint layout of PoseTrack21, which the pretrained SimpleBaseline estimator and the whole-body autoencoder use. The detector keeps the person class of COCO with a score of 0.7 or higher. It differs from the YOLOX of the SAMIDARE paper, so a session does not reproduce the numbers of that paper.

## Setup

The tools need Linux on x86_64, [uv](https://docs.astral.sh/uv/) and one NVIDIA GPU for tracking. The GPU driver has to support CUDA 12.4, which means that the header of `nvidia-smi` shows 12.4 or higher. Learning runs on the same GPU unless you choose the CPU.

Run the setup, the weight installation and the check from the root of the repository.

```bash
bash annotation/setup.sh
annotation/.venv/bin/vatl4pose-download-weights --weights-root pretrained_models
annotation/.venv/bin/vatl4pose-doctor paths.weights_root=pretrained_models
```

`setup.sh` creates `annotation/.venv` with Python 3.12 and leaves other environments unchanged. It fetches SAMIDARE and SAM 2 into `annotation/.deps/` at the commits pinned in `upstream.json` (SAMIDARE `4e663fa3`, SAM 2 `2b90b9f5`) and stops when a checkout there is at another commit or has local changes. It installs the packages of `requirements.lock` (torch 2.6.0 and torchvision 0.21.0 for CUDA 12.4), the package of this repository from its root in editable mode, and SAM 2 without its optional CUDA extension. It downloads the SAM 2.1 large checkpoint and verifies it against the SHA-256 in `models.json`, then downloads the COCO weights of the torchvision Faster R-CNN ResNet50 FPN v2. The commits that were installed go to `annotation/.deps/upstream_commits.json`. The repository contains no source code of SAMIDARE (MIT License), SAM 2 (Apache 2.0) or Label Studio (Apache 2.0). The License section of the README at the root of the repository describes the terms of the weights.

A session reads the SimpleBaseline estimator and the whole-body autoencoder of PoseTrack21 from `pretrained_models/` in the root of the repository. `vatl4pose-download-weights` installs the six weight files of the package, and `--from-file PATH` installs them from a zip that you downloaded yourself. The first session installs Label Studio 1.23.2 into `annotation/.venv-labelstudio` with uv. `vatl4pose annotate` runs `setup.sh` itself when no setup has completed or `requirements.lock` has changed since the last one. Git ignores the environments, `annotation/.deps/`, `annotation/samples/`, the runs and the session outputs.

## Run a session

Start one session from the root of the repository.

```bash
annotation/.venv/bin/vatl4pose annotate /path/to/video.mp4
```

The command prints the URL of the page, `http://localhost:8765` by default. `annotation/app.py` is the session itself and takes the same arguments, except for `--sample` and `--tools` of `vatl4pose annotate`. These are the options.

- `--output DIR` sets the session directory. The default is `outputs/annotation/<video name>-<hash>/` below the current directory, where the hash is the first 12 hex digits of the SHA-256 of the video.
- `--gpu N` picks the physical GPU. Without it the tool picks an idle GPU and waits for one.
- `--port N` sets the port of the page, 8765 by default.
- `--max-frames N` tracks only the first N frames. SAM 2 offloads the frames and its state to CPU memory, so memory use grows with the number of frames, and a short clip is a good first test.
- `--learning-device cuda|cpu` sets the device of the retraining and the selection of people, `cuda` by default. With `cpu` the GPU is used for tracking only, and a round in which you moved joints takes minutes instead of seconds (see Measured effort). Choose the device at the first start, because a resume keeps it.
- `--target-oks θ` turns on the stopping criterion of the paper with an accuracy θ, an OKS in (0, 1).
- `--tools PATH` (`vatl4pose annotate` only) names the `annotation/` folder. The command finds `annotation/` of the repository by itself, and `VATL4POSE_ANNOTATION_ROOT` sets it as well.
- `--sample` (`vatl4pose annotate` only) uses the local file `annotation/samples/people.mp4` instead of a video path. The repository does not contain this clip and git ignores `annotation/samples/`, so copy a short video of your own to that path before you use `--sample`.

The page lists the preparation steps and marks the current one: the GPU check, tracking with SAMIDARE with its frame count, preparing the data, training, selecting the next people and opening the editor. Keep the page open. The input video is not modified.

## A round

The editor shows one task per selected person with the person's box locked and the predicted keypoints. Drag a point onto its joint. To remove a hidden joint, select it and press Backspace. To add a missing point, pick its name below the image and click. Submit or Update saves a task. The editor has no Skip button, because Train needs every task saved. The top bar shows how many tasks are saved and enables Train once all of them are. The right column plays the SAMIDARE track of the person you annotate, from the first to the last frame of the track, with a thick box at the frame of the task. The link "All tracks" opens a video with every track.

The tools cannot edit boxes or track IDs, and the tracking stays fixed for the session. Duplicate boxes, missed people and split tracks can occur in crowded scenes, so check the video of all tracks before you annotate many people.

The cumulative share of annotated people follows `loop.query_ratio` of the package, which is 5, 10, 15, 20, 30, 40, 60, 80 and 100 % by default. The loop counts a person once per frame, so the first round asks for 5 % of the people.

When you press Train, the session exports the answers from Label Studio, checks that every task has a saved answer and writes them as the answers of the round. It retrains the estimator on them, writes the estimator, selects the next people and renders the results video. The page then shows the results of the round. The left side plays a video of the keypoints of everyone in the clip under the retrained estimator. The right side shows the round, the path of `estimator.pth` and two buttons, "Start round N" and "Finish here". When the loop has labeled everybody, the page reports that the session is complete.

## Stopping criterion

`--target-oks θ` adds the stopping criterion of the paper (SC_All, Sec. 3.3). After each round the loop checks whether the OKS of every person annotated so far, the people of this round included, reaches θ under the estimator before it learned from the round. People whose label marks no joint as visible are not counted.

The results page shows a verdict with the number of annotated people above θ and the lowest OKS. It also shows a table with one row per round: the share of labeled people, the number of people above θ out of all annotated people, and the lowest OKS. The editor page shows the same table in its right column. When the criterion is met, the page proposes "Finish" and offers "Keep annotating". Otherwise it proposes the next round and offers "Finish here". The loop never stops by itself, so you decide when to stop.

θ also sets which labeled people the retraining keeps (`retrain.finish_acc`, Eq. 3 of the paper). The retraining uses the people of the round and the earlier labeled people whose OKS under the estimator before the retraining is at most θ + 0.05. Without `--target-oks`, the page shows no criterion and the retraining keeps all labeled people.

## Outputs and resuming

The session directory holds these files, with `RUN` standing for `atl/tracked/simplebaseline/THC+WPU_Coresetfilter/clip`.

- `RUN/estimator.pth` is the estimator after the last finished round. The loop overwrites it after every round.
- `RUN/oracle/round_XX_queries.json` and `round_XX_answers.json` hold the selected people and your answers of each round.
- `RUN/log.jsonl` has one line per round. With `--target-oks`, its `stopping` field holds the judgment (`theta`, `people`, `above`, `lowest`, `met`). The loop writes `result.json` with the same judgments after its last round.
- `prepared/` holds the tracking output: `dataset/` (the frames and `annotations/clip.json`), `sam2mot_results.txt` and `tracking.mp4`.
- `label-studio/` holds the Label Studio database and its `server.log`.
- `session.json` records the video and the settings, and `finished.json` records that you finished.

The `vatl4pose-*` commands load the estimator with `estimator=simplebaseline estimator.weights=/absolute/path/to/estimator.pth`.

Press Ctrl-C in the terminal to stop a session. The same command resumes it at the same point. It reuses the tracking and the saved tasks, returns to the results page of a trained round and shows the results of a finished session. A resume needs the same video and the same `--max-frames`, `--learning-device` and `--target-oks`, and the same tool and weight files, as `session.json` records them. Otherwise the session stops and asks for a new `--output`. `--gpu` and `--port` may differ.

After a hard kill, the next start recovers the session only when it can show that the old owner on the same host is dead. A GPU lease or a video lock that the kill left behind stays until you remove it. The section "Managing command execution" of the public README describes how to check its owner.

## GPU

Tracking needs one CUDA GPU. The tool takes a lease on it, which is a lock file below `~/.cache/vatl4pose/gpu-leases` (`VATL4POSE_GPU_LEASE_ROOT` selects another directory), and checks once that no other process uses the GPU. It does not wait three minutes for a second check, which `vatl4pose run` and `vatl4pose gpu-run` still do. Without `--gpu` the tool takes the first GPU that has no compute process and no lease. When none is free, the page shows "Waiting for a free GPU" and the tool checks again every 10 seconds. If the GPU that `--gpu` names is in use, the page shows the error, and you run the same command again. With `--learning-device cuda`, the tool takes the lease again in every round and checks once. With `cpu`, it releases the GPU after tracking.

## Remote access

The page answers only requests to `localhost` or `127.0.0.1` at the port of the session. On a remote server, forward the same port number from your computer and open `http://localhost:8765`.

```bash
ssh -L 8765:localhost:8765 user@server
```

The session prints this command when it starts. Label Studio runs on its own local port behind the page and needs no forwarding. When 8765 is in use, start the session with another `--port` and forward that number.

## Measured effort

The measurements of this paragraph come from sessions with `--learning-device cuda` on a GPU that another training job loaded to about 90 %, so they are not times of a free GPU. On a 40-frame clip with 277 people, the rounds asked for 13 or 14 people, and correcting one person took a median of 35 seconds, taken from the Label Studio lead times of two rounds. Tracking took about 3 seconds per frame at 1280x720. From pressing Train to the results page took 14 to 15 seconds when the predictions were saved unchanged, and 34 to 52 seconds when the answers moved the joints and the retraining ran.

The learning device sets most of the wait after Train. On the first 30 frames of a 1280x720 clip, the first round asked for 12 people, and every joint of them was moved. From pressing Train to the results page took 24 seconds with the default `cuda` on a free GPU, and 15.5 minutes with `--learning-device cpu` on a 32-core CPU that other jobs kept busy.

## Advanced use of the manual cycle

The steps of a round also run as separate commands. `prepare_video.py` tracks one video on a GPU.

```bash
annotation/.venv/bin/python annotation/prepare_video.py \
  --video /path/to/video.mp4 --output annotation/runs/my-video --gpu GPU_ID
```

`GPU_ID` is the index of an idle GPU, and `--max-frames N` limits the clip. The script reuses a finished output with the same video and settings and stops on any other existing output. The output holds `dataset/`, `sam2mot_results.txt` and `tracking.mp4`, which draws the boxes and IDs. To start from tracking results of your own, `mot_to_coco.py` converts a MOT text file and frames named `000001.jpg`, `000002.jpg` and so on.

```bash
annotation/.venv/bin/python annotation/mot_to_coco.py \
  --tracks /path/to/sam2mot_results.txt --images /path/to/img1 \
  --output annotation/runs/imported/dataset --video-id clip
```

`cycle.sh` runs one step of the loop on such a folder.

```bash
bash annotation/cycle.sh annotation/runs/my-video atl [GPU_ID]
bash annotation/cycle.sh annotation/runs/my-video annotate
bash annotation/cycle.sh annotation/runs/my-video answers /path/to/ls_export.json
bash annotation/cycle.sh annotation/runs/my-video stop
```

`atl` prepares the cache until it exists and runs `vatl4pose-atl` with SimpleBaseline, `strategy=thc_wpu_duw` and the weights of a session. It selects the people of the first round, and in a later call it reads the answers, retrains and selects again. It runs on the CPU unless you append a GPU index. It writes `estimator.pth` after every round and `final_estimator.pth` at the end, and it has no `--target-oks`. `annotate` starts Label Studio at `http://localhost:8765` (`LS_PORT` changes the port), creates a project for the newest round without answers and imports its tasks. Log in with the local account `demo@example.com` and the password `demo-password-2026`. Correct the points, save every task with Update, then export JSON from Label Studio. The COCO export renumbers the annotation IDs and cannot be used. `answers` converts the export into `round_XX_answers.json`, `stop` stops Label Studio, and the next `atl` starts the next round.

`workflow.example.yaml` shows how `vatl4pose run` drives the `atl` step. Its input paths point to a dataset that the repository does not contain, so replace them with your own.
