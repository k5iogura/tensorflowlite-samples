import os,sys
import json
import numpy as np
from pdb import *
import cv2

from tfjsonx import graph

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-j',"--json",  type=chF, default='detect.json')
    args.add_argument('-i',"--image", type=chF, default='dog.jpg')
    args = args.parse_args()

    print("** Setup")
    g=graph(args.json)
    g.allocate_graph()
    print("** Done\n** Invoke inference")
    questions=100
    questions=1
    corrects =0
    for i in range(questions):
        #number_img = mnist.test.images[0].reshape(-1,28,28,1)
        #number_gt  = mnist.test.labels[0]
        img = cv2.imread(args.image)
        set_trace()
        g.tensors[g.inputs[0]].set(img.reshape(1,300,300,3))
        y = g.invoke(verbose=False)


