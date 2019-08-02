#!/bin/bash

rm -rf build

cmake -Bbuild -H. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSANITIZE_ADDRESS=On \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DFLATBUFFERS_INCLUDE_DIR=$HOME//tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
  -DTFLITE_INCLUDE_DIR=$HOME//tensorflow/ \
  -DTFLITE_LIBRARY_DIR=$HOME//tensorflow/tensorflow/lite/tools/make/build
