#!/bin/bash -v
TENSORFLOW="../../tensorflow"
if [ ! -d tensorflow ];then
    if [ ! -d ../../tensorflow/tensorflow ];then
        echo submodule update
        pushd ../../
        git submodule init
        git submodule update
        popd
    fi
    if [ -d ../../tensorflow/tensorflow ];then
        echo link tensorflow to current
        ln -s ../../tensorflow/tensorflow .
    fi
fi
if [ ! -d ./tensorflow/lite/tools/make/downloads/flatbuffers ];then
    ./tensorflow/lite/tools/make/download_dependencies.sh
fi
./tensorflow/lite/tools/make/build_lib.sh
cp ../../tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a .

gcc main.cpp -I. -I./tensorflow -I./tensorflow/lite/tools/make/downloads -I./tensorflow/lite/tools/make/downloads/eigen -I./tensorflow/lite/tools/make/downloads/absl -I./tensorflow/lite/tools/make/downloads/gemmlowp -I./tensorlow/lite/tools/make/downloads/neon_2_sse -I./tensorflow/lite/tools/make/downloads/farmhash/src -I./tensorflow/lite/tools/make/downloads/flatbuffers/include -std=c++11 -lstdc++ -ltensorflow-lite -L./ -lm -pthread `pkg-config --libs opencv`

