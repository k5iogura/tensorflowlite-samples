# -*- coding: utf-8 -*-
import os, sys, re
import numpy as np
from   pdb import set_trace
from   inspect import getmembers

import struct

import tflite
from   tflite.Model import Model
import tflite.BuiltinOptions
import tflite.TensorType

import tflite.AddOptions
import tflite.CallOptions
import tflite.ConcatenationOptions
import tflite.Conv2DOptions
import tflite.DepthwiseConv2DOptions
import tflite.FullyConnectedOptions
import tflite.L2NormOptions
import tflite.Pool2DOptions
import tflite.QuantizationParameters
import tflite.RNNOptions
import tflite.ReshapeOptions
import tflite.ResizeBilinearOptions
import tflite.SoftmaxOptions

import tflite.OperatorCode
import tflite.BuiltinOperator
import tflite.ActivationFunctionType
from   flags import flags

import cv2

from   fbnnop import DEPTHWISE_CONV_2D, MAX_POOL_2D, CONV_2D, RELUx
#from   fbnnpp import *
from   fbapix import *

if __name__=='__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-t',"--tflite",       type=chF, default='mnist.tflite')
    args.add_argument('-i',"--images",       type=int, default=1)
    args.add_argument('-q',"--quantization", action='store_true')
    args.add_argument('-v',"--verbose",      action='store_true')
    args = args.parse_args()
    if args.quantization:
        print("Inference with UINT8 Quantization")
        flags.floating_infer = False
    else:
        print("Inference with Default type")

    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    g = graph(tflite=args.tflite, verbose=args.verbose)
    g.allocate_graph(verbose=True)

    corrects = 0
    for i in range(args.images):
        
        number_img = mnist.test.images[i]
        number_gt  = mnist.test.labels[i]
        # input-type inference-type
        # uint8      uint8           no-convert
        # uint8      float           convert
        # float      uint8           NG
        # float      float           no-convert
        if args.quantization:
            assert g.tensors[g.inputs[0]].type == 'UINT8',"-q {} but input {}".format(args.quantization, g.tensors[g.inputs[0]].type)
            #g.tensors[g.inputs[0]].set((255*number_img[np.newaxis,:]).astype(np.uint8))
            g.tensors[g.inputs[0]].set((255*number_img[np.newaxis,:]).astype(np.uint8))
        else:
            g.tensors[g.inputs[0]].set(number_img[np.newaxis,:].astype(np.float32))
        y = g.invoke(verbose=False)
        gt = np.argmax(number_gt)
        pr = np.argmax(y)
        if gt!=pr:
            print("{:5d} incorrenct:gt-{} pr-{}".format(i,gt,pr))
        else:
            corrects+=1

    print("accurracy %.3f %d/%d"%(1.0*corrects/args.images,corrects,args.images))

