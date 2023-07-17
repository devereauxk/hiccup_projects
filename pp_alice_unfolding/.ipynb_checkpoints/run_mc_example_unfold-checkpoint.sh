#!/bin/bash

# PYJETTY_DIR should already be set to '/global/cfs/cdirs/alice/kdevero/mypyjetty/pyjetty' (where ever pyjetty was installed)
INFILE="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/569/LHC18b8_fast/10/282304/0004/AnalysisResults.root"
OUTPUT_DIR="./output"

python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_mc_ENC.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_pp.yaml -f $INFILE -o $OUTPUT_DIR
