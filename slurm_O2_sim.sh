#!/bin/bash
#SBATCH --job-name=O2_sim
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --output=/home/kdevereaux/hiccup_projects/slurm-%j.out

source /home/kdevereaux/env.txt

#run the application:
srun /home/kdevereaux/hiccup_projects/process_O2_sim.sh
