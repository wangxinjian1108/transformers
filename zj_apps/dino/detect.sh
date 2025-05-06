#!/bin/bash

img_path=${1:-"tmp/cone_side_right.png"}
device=${2:-"cpu"}
output_path=${3:-"tmp/cone_output.png"}

set -e

# 2. run the script
python xj_projects/dino/detect_bbox.py \
    --image $img_path \
    --prompts "person" "car" "suv" "van" "bus" "truck" "motorcycle" "bicycle" "cone" \
    --device $device \
    --output $output_path
