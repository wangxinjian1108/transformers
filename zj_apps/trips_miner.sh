#!/bin/bash
yaml_file=${1:-zj_apps/cfg/trips_miner_202503.yaml}

if [ -f /.dockerenv ]; then
    source /opt/ros/noetic/setup.sh
    source /opt/plusai/setup.sh
    python zj_apps/data_miner/trips_miner.py --config_yaml $yaml_file
    exit 0
fi

source ./zj_apps/env_setup.sh

docker exec $VAUTO_CONTAINER /bin/bash -c "source /opt/ros/noetic/setup.sh && source /opt/plusai/setup.sh && \
    python zj_apps/data_miner/trips_miner.py --config_yaml $yaml_file"

# python zj_apps/data_miner/trips_miner.py --config_yaml zj_apps/cfg/trips_miner_202503.yaml
