#!/bin/bash -v
tflite_convert --output_file=mnist_cnn.tflite --graph_def_file=mnist_frozen.pb --inference_type=QUANTIZED_UINT8 --inference_input_type QUANTIZED_UINT8 --input_arrays inputX --output_arrays outputX --std_dev_values 127 --mean_values 30 --default_ranges_min=-50 --default_ranges_max=255
