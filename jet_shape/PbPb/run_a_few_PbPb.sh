#!/bin/bash

# Input file containing the list of files
FILE_LIST="/global/cfs/cdirs/alice/wenqing/ENC/files_LHC18qr.txt"

# Directory to store the output files
OUTPUT_DIR="./output_data/temp"

# Number of iterations
N=10  # Set your desired number of iterations

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop N times
for ((counter=1; counter<=N; counter++)); do
    # Read the line from the file list
    INFILE=$(sed -n "${counter}p" "$FILE_LIST")

    # Skip empty lines
    [ -z "$INFILE" ] && continue

    # Run the Python script with the current input file
    python "$PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_data_jet-trk.py" \
        -c "$PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_PbPb.yaml" \
        -f "$INFILE" \
        -o "$OUTPUT_DIR/AnalysisResults_$counter/"
done

bash corral_files.sh $OUTPUT_DIR

hadd -j $OUTPUT_DIR/merged.root $OUTPUT_DIR/*.root
rm $OUTPUT_DIR/AnalysisResults* 
