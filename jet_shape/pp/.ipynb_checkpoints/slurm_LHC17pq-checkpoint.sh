#! /bin/bash

#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=kyle17pq
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=1-500
#SBATCH --output=/global/cfs/projectdirs/alice/kdevero/jobout/slurm-%A/%a.out

FILE_PATHS='/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC17pq/448/files.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 500 + 1 ))
echo "Files per job: $FILES_PER_JOB"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START=$START"
echo "STOP=$STOP"

for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  FILE="/global/cfs/projectdirs/alice/alicepro/hiccup${FILE}"
  bash process_LHC17pq.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
done
