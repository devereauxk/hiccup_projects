#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <WORKING_DIR>"
    exit 1
fi

WORKING_DIR="$1"
OUTPUT_DIR="$WORKING_DIR"

# Loop through all subdirectories in WORKING_DIR
for subdir in "$WORKING_DIR"/*/; do
    subdir_name=$(basename "$subdir")
    echo "processing subdirectory: $subdir_name"
    mv "$subdir/AnalysisResults.root" "$OUTPUT_DIR/AnalysisResults_${subdir_name}.root"
    if [ -f "$subdir/preprocessed_data.root" ]; then
        mv "$subdir/preprocessed_data.root" "$OUTPUT_DIR/preprocessed_data_${subdir_name}.root"
    fi
    if [ -f "$subdir/preprocessed_mc.root" ]; then
        mv "$subdir/preprocessed_mc.root" "$OUTPUT_DIR/preprocessed_mc_${subdir_name}.root"
    fi
    rm -r $subdir
done
