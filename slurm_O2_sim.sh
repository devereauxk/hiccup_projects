#!/bin/bash
#SBATCH --job-name=O2_sim
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=02:00:00
#SBATCH --output=/home/kdevereaux/hiccup_projects/slurm-%j.out

#run the application:
bash /home/kdevereaux/hiccup_projects/process_O2_sim.sh
