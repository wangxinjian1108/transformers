#!/bin/bash

img_dir=${1:-"tmp/cones"}
prompts=${2:-"person, car, suv, van, bus, truck, motorcycle, bicycle, cone"}
output_dir=${3:-"tmp/cones_output"}

rm -rf $output_dir
mkdir -p $output_dir

python xj_projects/dino/detect_img_dirs.py \
    --img-dir $img_dir \
    --prompts $prompts \
    --output-dir $output_dir \
    --model IDEA-Research/grounding-dino-base
