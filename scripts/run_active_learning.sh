#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=$1 # set GPU device

CONFIG="configs/al_simple.yaml" # config file
UNCERTAINY="THC+WPU" # Option: None HP TPC THC_L1 THC_L2 WPU MPE Margin Entropy
REPRESENTATIVENESS="None" # Option: None Random Influence
FILTER="Coreset" # Option: None Random Diversity K-Means weighted Coreset
MEMO="WACV_DUW$3" # memo for the experiment
# VIDEO_ID_LIST="configs/PCIT_video_exlist.txt" # list of video ids (e.g. 000342, 000522,...)
# VIDEO_ID_LIST="configs/posetrack_8split/val_video_list_full$2.txt" # PCIT video ids (e.g. 004, 007,...)
VIDEO_ID_LIST="configs/val_video_list_full$2.txt" # PCIT video ids (e.g. 004, 007,...)
VIDEO_LIST=$(cat ${VIDEO_ID_LIST}) # read the video id list
WUNC=$3

for VIDEO_ID in ${VIDEO_LIST}; do # loop over the video id list
    echo "Video ID: ${VIDEO_ID}"
    python scripts/Run_active_learning.py \
        --cfg ${CONFIG} \
        --uncertainty ${UNCERTAINY} \
        --representativeness ${REPRESENTATIVENESS} \
        --filter ${FILTER} \
        --video_id ${VIDEO_ID} \
        --memo ${MEMO} \
        --seedfix \
        --continual \
        --wunc ${WUNC}
        # --PCIT
        # --optimize
done # end of for loop
echo "All the scripts are finished!"

# Path: scripts/run_active_learning.sh