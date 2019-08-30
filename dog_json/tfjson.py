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
        img = cv2.imread(args.image)
        img = cv2.resize(img, (300,300))/255.
        img = img[np.newaxis,:]
        g.tensors[g.inputs[0]].set(img)
        y = g.invoke(verbose=False)


