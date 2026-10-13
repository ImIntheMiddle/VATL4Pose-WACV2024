# Managing command execution

The existing `vatl4pose-*` commands remain available. The `vatl4pose` entry point adds a sequential workflow runner and read-only status inspection:

```bash
vatl4pose run --config workflow.yaml --gpu 2
vatl4pose status outputs/my_video --json
vatl4pose resume outputs/my_video
vatl4pose resume outputs/my_video --only-failed --gpu 4
```

Activate the environment that provides the CLI executables, or use `uv run`. Paths in a workflow resolve relative to its YAML file. Commands run as argument lists, without a shell. `videos` belongs to a job and expands `{video_id}` in its arguments and dependencies. `{run_dir}` expands to the absolute workflow directory. `{gpu_id}` is available to jobs with `gpu: true`.

```yaml
schema: vatl4pose.workflow/1
run_dir: outputs/my_video
inputs: []  # optional immutable files, such as checkpoints or source videos
jobs:
  - id: prepare
    videos: [clip]
    argv:
      - vatl4pose-prepare
      - dataset=coco_custom
      - dataset.root=/path/to/dataset
      - dataset.keypoints=posetrack21
      - "dataset.ids=[{video_id}]"
      - "paths.output_root={run_dir}/artifacts"
      - run.name=my_video
      - run.device=cpu
  - id: atl
    videos: [clip]
    depends_on: ["prepare.{video_id}"]
    gpu: true
    argv:
      - vatl4pose-atl
      - dataset=coco_custom
      - dataset.root=/path/to/dataset
      - dataset.keypoints=posetrack21
      - "dataset.ids=[{video_id}]"
      - "paths.output_root={run_dir}/artifacts"
      - paths.weights_root=/path/to/pretrained_models
      - run.name=my_video
      - run.device=cuda
      - estimator=simplebaseline
      - estimator.weights=posetrack21/simplebaseline_256x192.pth
      - ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth
      - strategy=thc_wpu_duw
      - oracle=coco_file
      - loop.save_final_model=true
```

A job may name a dedicated environment's executable with an absolute path. The manifest records resolved commands, executable/script and declared input hashes, package source identity and Python distribution versions. Resume refuses changed commands, code, environments or immutable inputs. Annotation answers are mutable run outputs; do not put them in `inputs`. Completed jobs are skipped. Failed and stopped jobs get a new attempt directory and log; dependent jobs remain blocked until their prerequisites complete. `--only-failed` leaves annotation-waiting jobs alone; ordinary `resume` reads new answers and continues them. No automatic retries or changes to seeds, batch sizes or learning settings occur.

`.status.json` records running, waiting_annotation, completed, failed and stopped states. The inspector also reads old loop completion markers and unanswered queries. A process owner is verified by host, Linux boot, PID namespace and start identity; an owner that cannot be checked from this host/namespace is reported as unknown. Status inspection does not import torch or open images. Existing locks are never reclaimed automatically.

GPU jobs share an owner-checked UUID lease below `~/.cache/vatl4pose/gpu-leases`; `VATL4POSE_GPU_LEASE_ROOT` can select a shared directory for all participating commands. A new holder checks for compute processes twice, 180 seconds apart. A live ancestor may pass the lease to its sequential children. On interruption the runner stops the owned process group before releasing the GPU; uncertain cleanup preserves the lease for manual recovery. This coordinates participating commands of the same user, not unrelated launchers or a cluster scheduler. GPU jobs must perform foreground work and respect `CUDA_VISIBLE_DEVICES`, rather than detach a new session or select another physical GPU. Managed subprocess execution requires Linux `/proc`; status reading remains conservative on other hosts.

For a single foreground command:

```bash
vatl4pose gpu-run --gpu 2 --status-dir outputs/one_command -- \
  vatl4pose-atl experiment=wacv2024_posetrack21 run.device=cuda
```
