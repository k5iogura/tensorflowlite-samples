#!/bin/bash

if [ $# -eq 0 ];then
  CASE=1
else
  CASE=$1
fi

if [ ${CASE} -eq 1 ];then

tflite_convert \
--output_file detect.tflite \
--graph_def_file ssd_net_frozen.pb \
--inference_type QUANTIZED_UINT8 \
--std_dev_values 1 \
--mean_values 128 \
--inference_input_type QUANTIZED_UINT8 \
--input_shapes 300,300,3 \
--input_arrays Placeholder \
--default_ranges_max 255 \
--default_ranges_min 0 \
--output_arrays="ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

elif [ ${CASE} -eq 2 ];then

tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_input_type FLOAT \
--inference_type=FLOAT \
--mean_values=128 \
--std_dev_values=128 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--output_arrays="ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

elif [ ${CASE} -eq 3 ];then

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
--output_arrays="ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

elif [ ${CASE} -eq 4 ];then

tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_type=FLOAT \
--output_arrays="ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

exit

fi
