#!/bin/bash
# docker run -it --rm --net host -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority ubuntu:16.04 bash
# docker cp docker_tfpip.sh ContainerID:/docker_tfpip.sh
# chmod +x docker_tfpip.sh ; ./docker_tfpip.sh
# 
apt update && apt install -y openssh-server git curl vim vim-syntax-gtk ctags make cmake swig libjpeg-dev zlib1g-dev python3-dev python3-numpy python3 python python-pip python3-pip libopencv-dev feh python-numpy

pip install numpy opencv-python

git clone https://github.com/tensorflow/tensorflow; cd tensorflow; ./tensorflow/lite/tools/pip_package/build_pip_package.sh

pip install /tmp/tflite_pip/python/dist/tflite_runtime-1.14.0-cp27-cp27mu-linux_x86_64.whl



