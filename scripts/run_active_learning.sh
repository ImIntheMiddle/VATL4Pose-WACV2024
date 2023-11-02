#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=$1 # set GPU device

CONFIG="configs/posetrack21/al_simple_posetrack.yaml" # config file
# CONFIG="configs/jrdb-pose/al_simple_jrdb.yaml" # config file
UNCERTAINY="THC+WPU" # Option: None HP TPC THC_L1 THC_L2 WPU MPE Margin Entropy
REPRESENTATIVENESS="None" # Option: None Random Influence
FILTER="Coreset" # Option: None Random Diversity K-Means weighted Coreset
MEMO="WACV_without_transfer" # memo for the experiment
VIDEO_ID_LIST="configs/posetrack21/val_video_list_$2.txt" # PoseTrack21 video ids (e.g. 000342, 000522,...)
# VIDEO_ID_LIST="configs/jrdb-pose/test_ids.txt" # JRDB-Pose video ids (e.g. 00, 01,...)
VIDEO_LIST=$(cat ${VIDEO_ID_LIST}) # read the video id list

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
        # --from_scratch \
        # --wunc $3 # lambda value
        # --optimize
        # --vis \
        # --onebyone
        # --THCvsWPU ${2}
        # --fixed_lambda \
        # --stopping \
        # --retrain_thresh ${THRESH}
        # --PCIT
done # end of for loop
echo "All the scripts are finished!"

# Path: scripts/run_active_learning.sh