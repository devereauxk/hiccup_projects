#! /bin/bash

OUTDIR=/home/kdevereaux/hiccup_projects/ali_practice
cd $OUTDIR

source /home/kdevereaux/setup_O2_env.sh # full path of file containing appropriate variables

o2-sim -n 2 -m PIPE ITS -g pythia8pp -e TGeant4 -j 2
