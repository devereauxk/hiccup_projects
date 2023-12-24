#!/bin/bash

# PYJETTY_DIR should already be set to '/global/cfs/cdirs/alice/kdevero/mypyjetty/' (where ever pyjetty was installed)
INFILE="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18qr/147-148/148/child_1/0830/AnalysisResults.root"
OUTPUT_DIR="./output_data"

python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_data_jet-trk.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_PbPb.yaml -f $INFILE -o $OUTPUT_DIR