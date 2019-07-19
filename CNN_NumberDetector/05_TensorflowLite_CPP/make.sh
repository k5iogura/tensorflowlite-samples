#!/bin/bash -v
if [ ! -d tensorflow ];then
    echo link tensorflow
    ln -s ~/tensorflow/tensorflow .
fi
gcc main.cpp -I. -I./tensorflow -I./tensorflow/lite/tools/make/downloads -I./tensorflow/lite/tools/make/downloads/eigen -I./tensorflow/lite/tools/make/downloads/absl -I./tensorflow/lite/tools/make/downloads/gemmlowp -I./tensorlow/lite/tools/make/downloads/neon_2_sse -I./tensorflow/lite/tools/make/downloads/farmhash/src -I./tensorflow/lite/tools/make/downloads/flatbuffers/include -std=c++11 -lstdc++ -ltensorflow-lite -L./ -lm -pthread `pkg-config --libs opencv`
