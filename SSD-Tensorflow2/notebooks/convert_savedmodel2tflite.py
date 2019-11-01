import numpy as np
import os,sys
import tensorflow as tf
from random import random

saved_model_dir = './saved_model'
tflite_file="detect.tflite"
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]

def representative_dataset_gen():
    cnvs = np.asarray([random()*255 for i in range(300*300*3)],dtype=np.float32).reshape((1,300,300,3))
    for i in range(100):
        yield [cnvs]
        
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()
with open(tflite_file,"wb") as f:
    f.write(tflite_quant_model)
    print("done model write to",tflite_file)

#    return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_QuantizeModel(self, input_py_type, output_py_type, allow_float)
# RuntimeError: Quantization not yet supported for op: SQUARE

sys.exit(-1)

