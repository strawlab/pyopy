#!/bin/bash
#
# Naïve (can easily fail) renaming for matlab functions and scripts in a directory.
# For options:
#   matlabrenamer.bash
#
# Needs java.
#

pushd `dirname $0` > /dev/null
MY_DIR=`pwd`
popd > /dev/null

ARGS4J_JAR=${MY_DIR}/libs/args4j-2.0.9.jar
MREN_JAR=${MY_DIR}/libs/matlabrenamer.jar

java -cp $MREN_JAR:$ARGS4J_JAR matlabrenamer.MatlabRenamerCLI "$@"
