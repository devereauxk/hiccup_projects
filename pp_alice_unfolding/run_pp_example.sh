#!/bin/bash

PYJETTY_DIR="/home/software/users/wenqing/pyjetty/pyjetty/alice_analysis"
#INFILE="/rstorage/alice/data/LHC17pq/448/20-06-2020/448_20200619-0610/unmerged/child_1/2029/AnalysisResults.root"
INFILE="/rstorage/alice/data/LHC2018bdefghijklmnop/775/LHC18h/000288806/0120/AnalysisResults.root"
OUTPUT_DIR="./output"

#CONFIG_FILE=process_pp.yaml
CONFIG_FILE=process_pp_nominal.yaml
python $PYJETTY_DIR/process/user/wenqing/process_data_ENC.py -c $PYJETTY_DIR/config/ENC/pp/$CONFIG_FILE -f $INFILE -o $OUTPUT_DIR
