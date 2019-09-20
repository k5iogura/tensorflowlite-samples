#!/bin/bash -v
tflite_convert \
--graph_def_file mnist_frozen.pb \
--output_file=mnist.tflite \
--input_arrays=inputX \
--output_arrays=outputX,\
Reshape,\
Relu,\
MaxPool,\
Relu_1,\
MaxPool_1,\
Relu_2,\
add_3
