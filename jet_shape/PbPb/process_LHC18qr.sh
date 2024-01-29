#! /bin/bash

# This script takes an input file path as an argument, and runs a python script to 
# process the input file and write an output ROOT file.
# The main use is to give this script to a slurm script.

# Take two command line arguments -- (1) input file path, (2) output dir prefix
if [ "$1" != "" ]; then
  INPUT_FILE=$1
  echo "Input file: $INPUT_FILE"
else
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  OUTPUT_DIR=$2
  echo "output dir: $OUTPUT_DIR"
else 
  echo "Wrong command line arguments"
fi

# Load modules
workdir=/global/cfs/cdirs/alice/$USER/mypyjetty
module load python/3.11
source $workdir/pyjettyenv/bin/activate
module use /global/cfs/cdirs/alice/heppy_soft/yasp/software/modules
module use heppy
module load cmake gsl root/6.28.00 HepMC2/2.06.11 LHAPDF6/6.5.3 pcre2/default swig/4.1.1 HepMC3/3.2.5
module use $workdir/pyjetty/modules
module load pyjetty
module list
echo $PYJETTY_DIR

# Run python script
#python process/user/wenqing/process_data_ENC.py -c config/ENC/pp/process_pp.yaml -f $INPUT_FILE -o $OUTPUT_DIR
python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_data_jet-trk.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_PbPb.yaml -f $INPUT_FILE -o $OUTPUT_DIR
