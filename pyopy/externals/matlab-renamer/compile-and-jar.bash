#!/bin/bash
export CLASSPATH=libs/args4j-2.0.9.jar
mkdir -p out
find . -name "*.java" | xargs javac -g -cp ${CLASSPATH} -sourcepath src -d out
rm libs/matlabrenamer.jar
jar cvMf libs/matlabrenamer.jar -C out .
rm -Rf out
