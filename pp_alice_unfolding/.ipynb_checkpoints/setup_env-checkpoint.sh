#!/bin/bash

echo "compiling user's own version of pyjetty"
$HEPPY_DIR/scripts/pipenv_heppy.sh run $PYJETTY_DIR/pyjetty/cpptools/build.sh
module use /home/software/users/wenqing/pyjetty/modules
module load pyjetty/1.0
echo current pyjetty directory is $PYJETTY_DIR
