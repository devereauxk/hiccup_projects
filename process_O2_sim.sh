#! /bin/bash

OUTDIR=/home/kdevereaux/hiccup_projects/ali_practice
cd $OUTDIR

~/o2-sim -n 5 -m PIPE ITS -g pythia8pp -e TGeant4 -j 2