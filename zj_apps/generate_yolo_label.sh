#!/bin/bash

img_dir=${1:-"tmp/cones"}
prompts=${2:-"person, car, suv, van, bus, truck, motorcycle, bicycle, cone"}
output_dir=${3:-"tmp/cones_output"}

python zj_apps/dino/generate_yolo_label.py \
    --img-dir $img_dir \
    --output-dir $output_dir \
    --prompts $prompts \
    --box-threshold 0.35 \
    --text-threshold 0.25 \
    --fps 10 \
    --device cpu
