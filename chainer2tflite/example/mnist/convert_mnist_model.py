import sys

import logging
logger = logging.getLogger('chainer2tflite')
logger.setLevel(logging.DEBUG)

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import function
from chainer import function_node

from chainer import serializers
from chainer import Link, Chain, ChainList

sys.path.append("../../")
import chainer2tflite


# https://docs.chainer.org/en/stable/examples/train_loop.html
class MyNetwork(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


chainer.print_runtime_info()

model = MyNetwork()
chainer.serializers.load_npz('mnist.model', model)

# Must assign name with `input0`
# The converter interprets this name for input(Placeholder) tensor
# (batch, 28*28)
x = chainer.Variable(np.zeros((1, 28 * 28), dtype=np.float32), name='input0')

# for some reason, need to pass the array(or tuple) of variables
chainer2tflite.export(model, [x], "mnist.tflite")
