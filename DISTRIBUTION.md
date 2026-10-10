# Distribution scope

Checked on 2026-10-10. Source licensing, checkpoint permissions and dataset agreements are separate. Matching SHA256 and successful installation do not establish redistribution permission.

## Source

The v1 source and wheel include the project's MIT license. The bundled HRNet definition includes the upstream MIT license and `UPSTREAM.md` recording the source revision and changes. Retain both licenses when distributing.

Git history and the `wacv2024` tag contain legacy third-party code. The v1 MIT license does not license those files. Distributing repository history requires a separate review. [AlphaPose's license](https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE) restricts its software to noncommercial research and does not grant general redistribution permission. The v1 source archive and wheel exclude that legacy tree.

## Checkpoints

The six files in `src/vatl4pose/cli/weights_manifest.json` form an installation manifest, not a license grant. Redistribution remains unconfirmed. Do not attach the candidate weights zip to a public release or describe the weights as MIT licensed before confirming provenance, initialization permissions and dataset agreements.

| Files | Outstanding confirmation |
| --- | --- |
| `posetrack21/{simplebaseline,fastpose,hrnet}_256x192.pth` | Checkpoint/initialization provenance and PoseTrack21 agreement; HRNet source licensing alone does not license a checkpoint. |
| `jrdb/fastpose_256x192.pth` | Initialization provenance and JRDB/PoseTrack21 training permissions. |
| `wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth` | Distribution of a model trained using PoseTrack21 annotations. |
| `wholebodyAE/JRDB2022/Hybrid/WholeBodyAE_zdim4_v1.pth` | Distribution of a model trained using JRDB annotations. |

The [PoseTrack21 agreement](https://docs.google.com/document/d/1unxTYm2nVH1Qr7iYtgFzkzPbu042c1MLyZUP8Nb7-Fs/edit) restricts the dataset and derivative works to noncommercial use and prohibits dataset sharing. It does not expressly settle distribution of these checkpoints. The [JRDB site](https://jrdb.erc.monash.edu/) displays CC BY-NC-SA 3.0; confirm the actual dataset agreement and its application to trained models rather than treating a website footer as sufficient.

The default weights URL targets the planned v1 release. Until that asset is published with confirmed permissions, use an authorized local archive with `vatl4pose-download-weights --from-file PATH`. The remote download is not a verified public installation route.

## Optional tracking

The separate adapters fetch dependencies; they do not bundle external source, checkpoints or example videos.

| Dependency | Checked scope |
| --- | --- |
| [SAMIDARE](https://github.com/ZabuZabuZabu/SAMIDARE), `d46d4b82ca3ce0ab82f205908182bd4b6e373654` | No license file or grant found for the SAMIDARE script. Obtain permission or a published license before claiming redistribution or unrestricted use. External fetching does not resolve this. |
| [SAM2](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license), `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Upstream explicitly licenses model checkpoints and training/demo code under Apache 2.0. Preserve applicable licenses/notices if redistributing. The adapter downloads from upstream. |
| [torchvision pretrained models](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights) | Upstream states that model terms can depend on training datasets. The COCO Faster R-CNN checkpoint is downloaded from upstream and is not bundled. |

Exclude PoseTrack21 images/videos, annotations, tracking outputs and local Label Studio databases from release assets. Use generated samples or user-supplied videos for public examples.
