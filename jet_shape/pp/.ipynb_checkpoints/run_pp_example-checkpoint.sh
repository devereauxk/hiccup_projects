#!/bin/bash

# PYJETTY_DIR should already be set to '/global/cfs/cdirs/alice/kdevero/mypyjetty/' (where ever pyjetty was installed)
INFILE="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC2018bdefghijklmnop/775/LHC18h/000288806/0120/AnalysisResults.root"
OUTPUT_DIR="./output_data"

python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_data_ENC.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_pp_data.yaml -f $INFILE -o $OUTPUT_DIR