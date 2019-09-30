#!/bin/bash -v
tflite_convert \
--output_file=mnist_cnn.tflite \
--graph_def_file=mnist_frozen.pb \
--inference_type=QUANTIZED_UINT8 \
--inference_input_type QUANTIZED_UINT8 \
--input_arrays inputX \
--std_dev_values 127 \
--mean_values 30 \
--default_ranges_min=-50 \
--default_ranges_max=255 \
--output_arrays=\
outputX,\
Reshape,\
Relu,\
MaxPool,\
Relu_1,\
MaxPool_1,\
Relu_2,\
add_3,\
