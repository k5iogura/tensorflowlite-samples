#!/bin/bash

tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_type=FLOAT \
--mean_values=128 \
--std_dev_values=128 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--output_arrays="ExpandDims,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops
exit

tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_dev_values=128 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--output_arrays="ExpandDims,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops
exit

tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_type=FLOAT \
--output_arrays="ExpandDims,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

exit
