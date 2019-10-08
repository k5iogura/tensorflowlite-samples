# -*- coding: utf-8 -*-
from pdb import set_trace
import numpy as np
import sys,os
import tensorflow as tf

import argparse
args = argparse.ArgumentParser()
def chF(f): return f if os.path.exists(f) else sys.exit(-1)
args.add_argument("pbfile",           type=chF, default='frozen_inference_graph.pb')
args.add_argument('-v',"--verbose",   action='store_true')
args = args.parse_args()

PATH_TO_FROZEN_GRAPH = args.pbfile
print('PATH_TO_FROZEN_GRAPH:{}'.format(PATH_TO_FROZEN_GRAPH))

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
  with tf.Session() as sess:
    ops = tf.get_default_graph().get_operations()
    all_output_tensors = {tensor for op in ops for tensor in op.outputs}
    all_input_tensors  = {tensor for op in ops for tensor in op.inputs}
    all_output_names = {tensor.name for op in ops for tensor in op.outputs}
    all_input_names  = {tensor.name for op in ops for tensor in op.inputs}
    all_op_names     = {op.name for op in ops}
    only_l0_outputs  = { i for i in (all_output_names - all_input_names) if ':0' in i and '/' not in i }
    only_inputs      = { i for i in (all_input_names - all_output_names) }

    feedables        = {i.name for i in all_input_tensors if tf.get_default_graph().is_feedable(i) and '/' not in i.name}

print('only_l0_outputs  :{}'.format(only_l0_outputs))
print('only_inputs      :{}'.format(only_inputs))
print('feedable tensors :{}'.format(feedables))
