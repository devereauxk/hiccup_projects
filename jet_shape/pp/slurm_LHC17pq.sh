#! /bin/bash

#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=kyle17pq
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --array=1-20
#SBATCH --output=/global/cfs/projectdirs/alice/kdevero/jobout/slurm-%A/%a.out

FILE_PATHS='/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC17pq/448/files.txt'
NFILES=$(wc -l < $FILE_PATHS)
NFILES=200
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 20 + 1 ))
echo "Files per job: $FILES_PER_JOB"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START=$START"
echo "STOP=$STOP"

# Define output path from relevant sub-path of input file
OUTPUT_DIR="/global/cfs/cdirs/alice/kdevero/pp_jet-trk/$SLURM_ARRAY_JOB_ID"
echo "Output dir: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  echo "===================== PROCESSING FILE ${JOB_N} ====================="
  INPUT_FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  INPUT_FILE="/global/cfs/projectdirs/alice/alicepro/hiccup${INPUT_FILE}"
  bash process_LHC17pq.sh $INPUT_FILE $OUTPUT_DIR
  mv $OUTPUT_DIR/AnalysisResults.root $OUTPUT_DIR/AnalysisResults_${JOB_N}.root
done
