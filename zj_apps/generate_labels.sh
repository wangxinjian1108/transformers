#!/bin/bash

clips_txt=${1:-"/mnt/juicefs/xinjian/non_vehicle_data/clips_candidate_202503/cone_clips.txt"}
output_dir=${2:-"/mnt/juicefs/xinjian/non_vehicle_data/cone_data202503"}
prompts=${3:-"person, car, suv, van, bus, truck, motorcycle, bicycle, cone"}

python zj_apps/dino/generate_yolo_label_from_plusai_data.py \
    --clips-txt $clips_txt \
    --output-dir $output_dir \
    --prompts $prompts \
    --video-path cone_detect_202503.mp4 \
    --cameras side_right_camera \
    --device cpu \
    --check_indicator 0 \
    --interval_frames 2 \
    --object-nbs "cone,4,person,1"
