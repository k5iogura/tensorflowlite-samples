#!/bin/bash -v
tflite_convert \
--output_file=mnist_cnn.tflite \
--graph_def_file=mnist_frozen.pb \
--inference_type=QUANTIZED_UINT8 \
--inference_input_type QUANTIZED_UINT8 \
--input_arrays inputX \
--std_dev_values 1 \
--mean_values 127 \
--default_ranges_min=-50 \
--default_ranges_max=255 \
--output_arrays=\
outputX,\
MaxPool,\
MaxPool_1,\
Relu,\
Relu_1,\
Relu_2,\
Reshape,\
add_3
