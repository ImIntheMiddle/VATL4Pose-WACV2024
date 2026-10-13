# Distribution scope

Checked on 2026-10-13. The source, the weights and the datasets have separate terms.

## Source

The v1 source and wheel include the project's MIT license. The bundled HRNet definition includes the upstream MIT license and `UPSTREAM.md`, which records the source revision and every change. Keep both licenses when you distribute the source.

The Git history and the tag `wacv2024` hold the legacy third-party code of the paper version. The v1 MIT license does not cover those files, and their own licenses apply. [AlphaPose's license](https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE) restricts its software to noncommercial research. The v1 source archive and wheel exclude that legacy tree.

## Weights

The release `v1.0.0` holds the six files of `src/vatl4pose/cli/weights_manifest.json` in `vatl4pose-weights-v1.zip`, and `vatl4pose-download-weights` installs them.

| Files | Origin |
| --- | --- |
| `posetrack21/{simplebaseline,fastpose,hrnet}_256x192.pth`, `jrdb/fastpose_256x192.pth` | The estimators of the paper. The release `models` distributes the same files (same SHA256). |
| `wholebodyAE/{Posetrack21,JRDB2022}/Hybrid/WholeBodyAE_zdim4_v1.pth` | The whole-body autoencoders, retrained for v1 on the keypoint annotations of PoseTrack21 and JRDB-Pose. |

The weights are trained on PoseTrack21 and JRDB-Pose and are provided for noncommercial research only. The MIT license of the source does not cover them. Use them under the terms of the datasets. The [PoseTrack21 agreement](https://docs.google.com/document/d/1unxTYm2nVH1Qr7iYtgFzkzPbu042c1MLyZUP8Nb7-Fs/edit) restricts the dataset and derivative works to noncommercial use, and the [JRDB site](https://jrdb.erc.monash.edu/) gives CC BY-NC-SA 3.0.

Release assets exclude the images, videos and annotations of PoseTrack21 and JRDB-Pose.
