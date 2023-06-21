#!/bin/bash
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -J O2_sim
#SBATCH -t 02:00:00
#SBATCH --output=/home/kdevereaux/ali_practice/slurm-%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 256 /home/kdevereaux/hiccup_projects/process_O2_sim.sh
