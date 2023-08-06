#! /bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=kyle18b8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=1-2000
#SBATCH --output=/global/cfs/projectdirs/alice/kdevero/jobout/slurm-%A_%a.out

FILE_PATHS='/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/569/files.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 2000 + 1 ))
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
  srun -n 1 -c 1 process_LHC18b8.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
done
