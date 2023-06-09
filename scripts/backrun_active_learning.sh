#!/bin/bash
set -x

# set GPU device
CONFIG="configs/al_simple.yaml" # config file
UNCERTAINY="$1" # Option: None HP TPC THC_L1 THC_L2 WPU_hybrid WPU_raw MPE Margin Entropy
REPRESENTATIVENESS="$2" # Option: None Random Influence
FILTER="$3" # Option: None Random Diversity K-Means weighted
MEMO="PCIT3" # memo for the experiment
VIDEO_ID_LIST="configs/PCIT_video_exlist.txt" # list of video ids (e.g. 000342, 000522,...)
# VIDEO_ID_LIST="configs/val_video_list_ex1.txt" # PCIT video ids (e.g. 004, 007,...)
VIDEO_LIST=$(cat ${VIDEO_ID_LIST}) # read the video id list

GPU_NUM=$4 # number of GPUs
i=0 # for GPU device index
for VIDEO_ID in ${VIDEO_LIST}; do # loop over the video id list
    if [ $i -eq $GPU_NUM ]; then # if the GPU device index is larger than the number of GPUs
        wait # wait until all the scripts are finished
        i=0 # reset the GPU device index
    fi
    {
    echo "Video ID: ${VIDEO_ID}"
    GPU_INDEX=$(($i % $GPU_NUM)) # GPU device index
    export CUDA_VISIBLE_DEVICES=${GPU_INDEX} # set GPU device
    python scripts/Run_active_learning.py \
        --cfg ${CONFIG} \
        --uncertainty ${UNCERTAINY} \
        --representativeness ${REPRESENTATIVENESS} \
        --filter ${FILTER} \
        --video_id ${VIDEO_ID} \
        --memo ${MEMO} \
        --seedfix \
        --PCIT \
        --vis
        # --optimize
    } & # run the script in the background
    i=$(($i+1)) # increase the GPU device index
done # end of for loop
wait # wait until all the scripts are finished
echo "All the scripts are finished!"

# Path: scripts/run_active_learning.sh