#!/bin/bash
cd / ; git clone https://github.com/k5iogura/tensorflowlite-samples;cd tensorflowlite-samples

unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
python dog_detect_tflite_runtime.py
feh result.jpg
