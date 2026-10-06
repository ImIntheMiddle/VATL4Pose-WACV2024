# HRNet (pose_hrnet.py)

Source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch, file `lib/models/pose_hrnet.py`

Commit: `6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1` (master on 2026-10-06, committed 2022-12-14). The file itself last changed in `94750322734f5fa50756a4032e13913d40c10fec` (2020-02-24).

License: MIT, Copyright (c) 2019 Leo Xiao. `LICENSE` is the upstream file at the same commit, unchanged.

Used by `vatl4pose.models.estimator.HRNetEstimator`, which subclasses `PoseHighResolutionNet`. The package builds HRNet-W32 (stages 2, 3 and 4 as in the upstream `w32_256x192` setting) with a 1x1 output convolution.

## Local changes

Line numbers refer to upstream for removed code and to the vendored file for added or modified code. Everything else, including the blocks, the high-resolution module and `forward`, is identical to upstream.

| Upstream lines | Change |
| --- | --- |
| 11-12, 14, 19 | Removed `import os`, `import logging`, `import torch` and `logger = logging.getLogger(__name__)`. Nothing left in the file uses them. |
| 124, 130, 136 | Removed the three `logger.error(error_msg)` calls in `HighResolutionModule._check_branches`. The `raise ValueError(error_msg)` that follows each call stays. |
| 276-278 (vendored 268-269) | The constructor `__init__(self, cfg, **kwargs)` became `__init__(self, num_joints, stage2, stage3, stage4, final_conv_kernel=1)`. The line `extra = cfg['MODEL']['EXTRA']` is gone. |
| 291, 301, 312 (vendored 282, 292, 303) | `extra['STAGE2']`, `extra['STAGE3']` and `extra['STAGE4']` became the arguments `stage2`, `stage3` and `stage4`. |
| 325-328 (vendored 316-319) | `cfg['MODEL']['NUM_JOINTS']` became `num_joints`, and `extra['FINAL_CONV_KERNEL']` became `final_conv_kernel`, in the construction of `final_layer`. |
| 331 | Removed `self.pretrained_layers = extra['PRETRAINED_LAYERS']`. |
| 462-492 | Removed `PoseHighResolutionNet.init_weights`, which initialises the weights and loads a checkpoint file with `torch.load` and logging. The estimator wrapper loads checkpoints itself with `weights_only=True`. |
| 495-501 | Removed `get_pose_net`, which read a yacs-style config. |

`third_party/hrnet/__init__.py` is new and empty. It makes the directory importable as `third_party.hrnet`.
