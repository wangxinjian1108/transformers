#!/bin/bash
set -e

pull_docker_image_if_not_exists() {
    image_name=$1
    if [ -z "$(docker images -q $image_name 2>/dev/null)" ]; then
        echo "$image_name does not exist, pulling it..."
        docker pull $image_name
    fi
}

start_docker_container() {
    local CONTAINER_NAME=$1
    local IMAGE_NAME=$2
    local WORK_DIR=$3

    pull_docker_image_if_not_exists $IMAGE_NAME
    container_id=$(docker ps --format '{{.ID}} {{.Names}}' | awk -v cname="$CONTAINER_NAME" '$2 == cname {print $1}')

    if [ ! $container_id ]; then
        docker run -i -d --name $CONTAINER_NAME \
            -v $HOME/:$HOME/:rw \
            -v /mnt/:/mnt/:rw \
            -v /juicefs/:/juicefs/:rw \
            -v ~/:/root/ \
            -v $WORK_DIR/:$WORK_DIR/:rw \
            --user $(id -u):$(id -g) \
            --gpus all \
            --workdir $WORK_DIR \
            $IMAGE_NAME /bin/bash
        echo "container $CONTAINER_NAME started"
    fi
}

check_path_arg() {
    if [ ! -e "$1" ]; then
        echo "$1 does not exist"
        exit 1
    fi
}

check_path_args() {
    for path in "$@"; do
        if [ ! -e "$path" ]; then
            echo "$path does not exist"
            exit 1
        fi
    done
}

get_gpu_architecture() {
    local gpu_device
    gpu_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    if [[ $gpu_device == *"A100"* || $gpu_device == *"RTX 30"* ]]; then
        echo "Ampere86"
    else
        echo "Turing75"
    fi
}

# Function to get CUDA device information from nvidia-smi
function check_cuda_device {
    local min_memory_mb=$1

    # Use nvidia-smi to get the memory information of each GPU
    local nvidia_smi_output
    nvidia_smi_output=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    # Check if nvidia-smi returned any output
    if [ -z "$nvidia_smi_output" ]; then
        echo "No CUDA devices found or nvidia-smi command failed."
        return 1
    fi

    local best_device_id=-1
    local max_free_mem=0

    # Read each line from nvidia-smi output
    while IFS=',' read -r device_id free_mem; do
        free_mem=$(echo $free_mem | tr -d ' ') # Remove any spaces
        if [ "$free_mem" -ge "$min_memory_mb" ]; then
            if [ "$free_mem" -gt "$max_free_mem" ]; then
                max_free_mem=$free_mem
                best_device_id=$device_id
            fi
        fi
    done <<<"$nvidia_smi_output"

    if [ "$best_device_id" -eq -1 ]; then
        echo -1
        return 1
    fi

    echo $best_device_id
    return 0
}

# Main function to find the best device
function find_best_device {
    local min_memory_mb=$1

    check_cuda_device "$min_memory_mb"
    local result=$?
    if [ $result -ne 0 ]; then
        return 1
    fi
}

# Example usage:
# find_best_device 30000
