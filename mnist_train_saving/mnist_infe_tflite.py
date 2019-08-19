#!/usr/bin/env python3
import os, sys
import numpy as np
from pdb import *
#import cv2
from tflite_runtime import interpreter as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from time import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

ip = tf.Interpreter(model_path="./mnist.tflite")
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
questions= 1000
corrects = 0
for inferNo in range(questions):
    number_img, number_out = mnist.test.next_batch(1)
    ip.set_tensor(indexi, number_img)
    ip.invoke()
    gt = np.argmax(number_out)
    pd = np.argmax(ip.get_tensor(indexo))
    if gt != pd:
        if corrects<100: print("incorrect truth:prediction = %d : %d"%(gt,pd))
    else:
        corrects+=1
print("total %.3f(%d/%d)"%(1.0*corrects/questions,corrects,questions))
