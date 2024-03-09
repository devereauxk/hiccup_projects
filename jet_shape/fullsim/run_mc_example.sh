#!/bin/bash

date
INFILE="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/569/LHC18b8_fast/15/282366/0004/AnalysisResults.root"
OUTPUT_DIR="./output_mc"

python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_mc_jet-trk.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_PbPb.yaml -f $INFILE -o $OUTPUT_DIR
date