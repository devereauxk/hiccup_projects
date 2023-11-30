#!/bin/bash

OUT_DIR="./5020_PbPb_40-50"
mkdir $OUT_DIR

# Loop through all files matching the pattern
i=0
for input_file in /global/cfs/cdirs/alice/kdevero/jetscape_hiccup/5020_PbPb_40-50_0.30_2.0_1/JetscapeHadronListBin100_110_*; do
    # Define the output file name
    output_file="$OUT_DIR/100_110_out_${i}.root"

    # Run the Python script with the specified parameters
    python3 analyze_events.py --R0 0.4 --py-pthatmin 100 --jet_ptmin 100 --jet_ptmax 110 --input "$input_file" --output "$output_file"

    echo "Analysis completed for $input_file. Output saved to $output_file"
	((i++))
done

rm merged.root
hadd -j merged.root *.root