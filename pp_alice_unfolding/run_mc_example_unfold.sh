#!/bin/bash

#PYJETTY_DIR="/home/software/users/wenqing/pyjetty/pyjetty/alice_analysis"
INFILE="/rstorage/alice/data/LHC18b8/569/LHC18b8_fast/10/282304/0004/AnalysisResults.root"
OUTPUT_DIR="./output"

#python ./process_mc_ENC.py -c $PYJETTY_DIR/config/ENC/pp/process_pp.yaml -f $INFILE -o $OUTPUT_DIR

python ./process_mc_ENC.py -c ./process_pp.yaml -f $INFILE -o $OUTPUT_DIR
