# Active Learning For Human Pose Estimation

![Sample Visualization](sample_viz.png)

This repository aims to collect and standardize results for various active learning algorithms for human pose estimation. While the repository contains official implementations for three algorithms: `LearningLoss++, EGL++ and VL4Pose`, we also include our code for various other algorithms that report results for human pose estimation. Further, **we provide visualizations** for various active learning algorithms to provide better insights into their sampling process!

## Table of contents
1. [Installation](#installation)
2. [Organization](#organization)
3. [Algorithms](#algorithms)
4. [Code Execution](#execution)
5. [Publications](#publications)
6. [Citation](#citation)
7. [About Me :D](#about)


## Installation <a name="installation"></a>
1. Create a new environment with `conda create --name AL4Pose python=3.8`
2. Activate this environment: `conda activate AL4Pose`
3. Install via conda: `conda install -c pytorch -c conda-forge -c anaconda pytorch opencv albumentations matplotlib numpy umap-learn scipy scikit-learn scikit-image tensorboard pandas torchaudio torchvision pyyaml seaborn jupyter`
4. Install via pip: `pip install adjusttext`

## Organization <a name="organization"></a>

The repository contains two main folders, `code`and `data`. We have a separate folder `cached` which stores two files: `Stacked_HG_ValidationImageNames.txt` and `mpii_cache_16jnts.npy`. The former stores the names of those images which are used in the validation dataset in the original Stacked Hourglass paper (ECCV 2016), whereas the latter file stores the post-processed MPII dataset to avoid redundant processing every time the code is run.

Our framework currently supports two datasets: MPII and LSP/LSPET. These datasets can be downloaded from [MPII Link](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz), [LSP Link](http://sam.johnson.io/research/lsp.html) and [LSPET Link](http://sam.johnson.io/research/lspet.html). For each of these datasets copy-paste all the images `*.jpg` into `data/{mpii OR lsp OR lspet}/images/`


```bash
├── LICENSE
├── README.md
├── cached
│   ├── Stacked_HG_ValidationImageNames.txt
│   └── mpii_cache_16jnts.npy
├── code
│   ├── activelearning.py
│   ├── activelearning_viz.py
│   ├── autograd_hacks.py
│   ├── config.py
│   ├── configuration.yml
│   ├── dataloader.py
│   ├── evaluation.py
│   ├── main.py
│   ├── models
│   │   ├── auxiliary
│   │   │   └── AuxiliaryNet.py
│   │   ├── hrnet
│   │   │   └── pose_hrnet.py
│   │   ├── learning_loss
│   │   │   └── LearningLoss.py
│   │   └── stacked_hourglass
│   │       ├── StackedHourglass.py
│   │       └── layers.py
│   └── utils.py
├── data
│   ├── lsp
│   │   ├── README.txt
│   │   ├── images
│   │   ├── joints.mat
│   │   └── lsp_filenames.txt
│   ├── lspet
│   │   ├── README.txt
│   │   ├── images
│   │   ├── joints.mat
│   │   └── lspet_filenames.txt
│   └── mpii
│       ├── README.md
│       ├── images
│       ├── joints.mat
│       └── mpii_filenames.txt
└── results
```

Running `python main.py` in the `code` folder executes the code, with configurations specified in `configuration.yml`

## Algorithms <a name="algorithms"></a>

The following algorithms are implemented which have also reported results for human pose estimation (alphabetical order):
1. Aleatoric Uncertainty
> We use an extension of Kendall and Gal's NeurIPS 2017 paper [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html) for human pose estimation. Our implementation includes computing the argmax of the heatmap for each joint to obtain a two dimensional coordinate. This allows us to directly apply the formulation in the original paper to human pose estimation.
2. Core-Set
> Sener and Savarese in their ICML 2018 [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://arxiv.org/abs/1708.00489) provided theoretical results to support Core-Set for deep neural networks. In our implementation we perform pooling over the penultimate layer to obtain a vector encoding for each image and subsequently use it for k-Centre Greedy.
3. EGL++
> In our WACV 2022 [Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides of the Same Coin?](https://openaccess.thecvf.com/content/WACV2022/html/Shukla_Bayesian_Uncertainty_and_Expected_Gradient_Length_-_Regression_Two_Sides_WACV_2022_paper.html), we show that computing the expected gradient length a.k.a EGL (an active learning algorithm) is equivalent to estimating Bayesian uncertainty in computer vision. Further, we propose a modification EGL++ that extends EGL to human pose estimation.
NOTE: PyTorch no longer supports register_backward_hook() so future versions of PyTorch (possibly after 1.13) may require modification of code. 
4. LearningLoss++ / Learning Loss
> Yoo and Kweon in their CVPR 2019 paper proposed a method, [Learning Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf), for learning the possible 'loss' for an input sample, and showcased results for classification, object detection and pose estimation. In a subsequent improvement, our work [A Mathematical Analysis of Learning Loss for Active Learning in Regression](https://openaccess.thecvf.com/content/CVPR2021W/TCV/papers/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.pdf) provides theoretical insights into Learning Loss and provides a new framework, LearningLoss++, using these insights to modify Learning Loss. LearningLoss++ reports results for human pose estimation.
5. Multi-Peak Entropy
> Amongst the earliest algorithms, [Multi-Peak Entropy](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Active_Learning_for_ICCV_2017_paper.pdf) (ICCV 2017) leveraged spatial ambiguities in the heatmap to identify images for active learning. We expect that confident predictions will have a unimodal heatmap whereas images containing joint level ambiguities tend to have multiple modes. Multi-Peak entropy uses these ambiguities to measure entropy across various modes of the heatmap.

6. VL4Pose (Visual Likelihood for Pose Estimation)
> In our BMVC 2022 paper ([VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation](https://bmvc2022.mpi-inf.mpg.de/610/)) we propose VL4Pose; an algorithm that utilizes simple domain knowledge to unify joint and pose level uncertainty and perform fast yet reliable active learning, out-of-distribution and pose refinement (to a limited extent). VL4Pose does this by training an auxiliary network to maximize the likelihood of training poses. As a consequence, out-of-distribution poses incur a low likelihood in our framework making them suitable candidates for active learning.

## Code Execution <a name="execution"></a>
The code needs to be run each time for every active learning cycle and for every algorithm separately. All active learning algorithms require a pretrained model, which is the base model for subsequent active learning cycles. A sample configuration file to create this base model is available in `sample_configs/base_model.yml`. For active learning algorithms such as Multi-Peak entropy and Core-Set which do not use an auxiliary model, we re-run the code with configuration: `/sample_configs/non_auxiliary.yml`. For active learning algorithms which require an auxiliary network (LearningLoss++, VL4Pose, Aleatoric), we need to load the base model and train the auxiliary network with objective specific to the algorithm. This configuration is available in `sample_configs/train_auxiliary_from_base.yml`. Now that we have a trained auxiliary network corresponding to the base model, we can perform active learning these methods using the configuration `sample_configs/auxiliary.yml`. In fact, for subsequent active learning algorithms we don't need to run the code separately to train the auxiliary network since the new configuration explicitly specifies training both: the pose estimator and the auxiliary network (gradients stopped from flowing into the pose estimator) simultaneously.

Visualization: Once we have a pretrained model (and auxiliary model if applicable), it is possible to visualize the sampling process of various active learning algorithms by simply setting `activelearning_visualization: True` (line 48 in `configuration.yml`).

## Publications <a name="publications"></a>

This code has been used to support the following three of our works:

1. _"VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation"_, BMVC 2022
2. _"Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?"_, WACV 2022
3. _"A Mathematical Analysis Of Learning Loss For Active Learning In Regression"_, CVPR-W 2021

## Citation <a name="citation"></a>

If you found this code useful, please consider citing _all_ three publications (it doesn't cost anything) :D
Also, **_feedback is welcome!_**

Please contact me at _megh.shukla [at] epfl.ch_

```
@inproceedings{Shukla_2022_BMVC,
author    = {Megh Shukla and Roshan Roy and Pankaj Singh and Shuaib Ahmed and Alexandre Alahi},
title     = {VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0610.pdf}
}

@INPROCEEDINGS{9706805,
  author={Shukla, Megh},
  booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?}, 
  year={2022},
  volume={},
  number={},
  pages={2021-2030},
  doi={10.1109/WACV51458.2022.00208}}

@INPROCEEDINGS{9523037,
  author={Shukla, Megh and Ahmed, Shuaib},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={A Mathematical Analysis of Learning Loss for Active Learning in Regression}, 
  year={2021},
  volume={},
  number={},
  pages={3315-3323},
  doi={10.1109/CVPRW53098.2021.00370}}

```

## About Me :D <a name="about"></a>

I am a first year Ph.D. student in Electrical Engineering at EPFL supervised by Prof. Alexandre Alahi, [Visual Intelligence for Transportation Lab](https://www.epfl.ch/labs/vita/) . My research interests lie in uncertainty estimation, active learning and probabilistic modelling for keypoint estimation. These interests are peppered throughout my fledgling career, at Mercedes-Benz and now at EPFL. As a computer vision research engineer, I lead R&D in active learning, an academic research area with tangible business benefits. I take pride in not only providing theoretical and applied advancements in active learning [BMVC22, WACV22, CVPRW21], but also in engineering my research into the project's data pipelines. Taking research beyond publications and into production allowed me a holistic view of the research and development cycle which remains a defining moment in my career. 
