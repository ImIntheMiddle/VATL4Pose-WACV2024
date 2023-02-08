#!/bin/bash
set -x

# set GPU device
export CUDA_VISIBLE_DEVICES="$1"
CONFIG="configs/al_simple.yaml" # config file
UNCERTAINY="None" # Option: None HP TPC THC_L1 THC_L2 WPU_hybrid WPU_raw
REPRESENTATIVENESS="Random" # Option: None Random Influence
FILTER="None" # Option: None Random Diversity K-Means
MEMO="Scratch" # memo for the experiment
VIDEO_ID_LIST="configs/val_video_list_$2.txt" # list of video ids (e.g. 000342, 000522,...)

for VIDEO_ID in $(cat ${VIDEO_ID_LIST}); do
    echo "Video ID: ${VIDEO_ID}"
    python scripts/Run_active_learning.py \
        --cfg ${CONFIG} \
        --uncertainty ${UNCERTAINY} \
        --representativeness ${REPRESENTATIVENESS} \
        --filter ${FILTER} \
        --video_id ${VIDEO_ID} \
        --memo ${MEMO} \
        --seedfix \
        --vis
        # --onebyone
        # --verbose
        # --speedup
done

# Path: scripts/run_active_learning.sh