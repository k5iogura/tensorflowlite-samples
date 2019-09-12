#!/usr/bin/env python3
import os, sys
import numpy as np
from pdb import *
#import cv2
from tflite_runtime import interpreter as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from time import time

import argparse
args = argparse.ArgumentParser()
def chF(f): return f if os.path.exists(f) else sys.exit(-1)
args.add_argument('-t',"--tflite",       type=chF, default='mnist.tflite')
args.add_argument('-i',"--images",       type=int, default=1)
args.add_argument('-q',"--quantization", action='store_true')
args.add_argument('-v',"--verbose",      action='store_true')
args = args.parse_args()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

ip = tf.Interpreter(model_path=args.tflite)
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
questions= args.images
corrects = 0
for inferNo in range(questions):
    if args.quantization:
        number_img = (mnist.test.images[inferNo]*255).reshape(1,-1)
        ip.set_tensor(indexi, number_img.astype(np.uint8))
    else:
        number_img, number_out = mnist.test.next_batch(1)
        ip.set_tensor(indexi, number_img.astype(np.float32))
    number_out = mnist.test.labels[inferNo].reshape(1,-1)
    ip.invoke()
    gt = np.argmax(number_out)
    pd = np.argmax(ip.get_tensor(indexo))
    if gt != pd:
        if corrects<100: print("incorrect truth:prediction = %d : %d"%(gt,pd))
    else:
        corrects+=1
print("total %.3f(%d/%d)"%(1.0*corrects/questions,corrects,questions))
