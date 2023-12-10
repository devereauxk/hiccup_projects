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
  JOB_ID=$2
  echo "Job ID: $JOB_ID"
else 
  echo "Wrong command line arguments"
fi

if [ "$3" != "" ]; then
  TASK_ID=$3
  echo "Task ID: $TASK_ID"
else
  echo "Wrong command line arguments"
fi

# Define output path from relevant sub-path of input file
OUTPUT_PREFIX="/global/cfs/cdirs/alice/kdevero/PbPb_jet-trk/$JOB_ID"
# Note: suffix depends on file structure of input file -- need to edit appropriately for each dataset
OUTPUT_SUFFIX=$(echo $INPUT_FILE | cut -d/ -f5-11)
#echo $OUTPUT_SUFFIX
OUTPUT_DIR="$OUTPUT_PREFIX/$TASK_ID"
echo "Output dir: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

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
python $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_data_ENC.py -c $PYJETTY_DIR/pyjetty/alice_analysis/process/user/kyle/process_PbPb_R02.yaml -f $INPUT_FILE -o $OUTPUT_DIR


# Move stdout to appropriate folder
#mv /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/slurm-${JOB_ID}_${TASK_ID}.out /global/cfs/cdirs/alice/kdevero/pp_alice_unfolding/AnalysisResults/${JOB_ID}/
