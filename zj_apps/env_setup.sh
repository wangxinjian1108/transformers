#!/bin/bash
set -e
source ./zj_apps/utils.sh

DOCKER_IMAGE="docker.pluscn.cn:5050/plusai/vauto_labeler:$(get_gpu_architecture)"
VAUTO_CONTAINER="auto_labeling_transformers_$(get_gpu_architecture)_$(whoami)"

start_docker_container $VAUTO_CONTAINER $DOCKER_IMAGE $(pwd)
