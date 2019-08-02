#!/bin/bash -v
mkdir -p build
if [ ! -d tensorflow ];then
    ln -s ~/tensorflow/tensorflow .
fi

gcc mnist-2.cc mnist-loader.cc \
-o build/mnist \
-I. \
-I./tensorflow \
-I./tensorflow/lite/tools/make/downloads \
-I./tensorflow/lite/tools/make/downloads/eigen \
-I./tensorflow/lite/tools/make/downloads/absl \
-I./tensorflow/lite/tools/make/downloads/gemmlowp \
-I./tensorlow/lite/tools/make/downloads/neon_2_sse \
-I./tensorflow/lite/tools/make/downloads/farmhash/src \
-I./tensorflow/lite/tools/make/downloads/flatbuffers/include \
-ltensorflow-lite \
-std=c++11 \
-lstdc++ \
-L./ \
-lm \
-pthread
