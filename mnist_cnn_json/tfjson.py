import os,sys
import struct
import json
import numpy as np
from pdb import *

from tfjsonx import graph

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-j',"--json",       type=chF, default='detect.json')
    args.add_argument('-i',"--invoke_layer", type=int, default=0)
    args = args.parse_args()

    print("** Setup")
    g=graph(args.json)
    g.invoke_layer = args.invoke_layer
    g.allocate_graph()
    print("** Done\n** Invoke inference")
    questions=100
    questions=20
    questions=1
    questions=2
    corrects =0
    for i in range(questions):
        number_img = mnist.test.images[i]
        number_gt  = mnist.test.labels[i]
        g.tensors[g.inputs[0]].set(number_img[np.newaxis,:])
        y = g.invoke(verbose=False)
        gt = np.argmax(number_gt)
        pr = np.argmax(y)
        if gt!=pr:
            print("incorrenct:",gt,pr)
        else:
            corrects+=1

    print("accurracy %.3f %d/%d"%(1.0*corrects/questions,corrects,questions))

