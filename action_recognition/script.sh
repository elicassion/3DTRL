#!/usr/bin/env bash
source ~/.bashrc
export NCCL_IB_DISABLE=1
config=$1
data_loc=$2
output_loc=$3

conda activate timesformer

python tools/run_net.py --cfg configs/$config DATA.PATH_TO_DATA_DIR $data_loc OUTPUT_DIR $output_loc
