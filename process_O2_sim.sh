#! /bin/bash

OUTDIR=/home/kdevereaux/ali_practice/slurm
cd $OUTDIR

~/o2-sim -n 5 -m PIPE ITS -g pythia8pp -e TGeant4 -j 2