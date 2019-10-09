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


class graph():
    def __init__(self, detection_graph):
        with detection_graph.as_default():
          with tf.Session() as sess:
            self.operators = ops = tf.get_default_graph().get_operations()
            self.operators = ops = [ o for o in ops if 'Variable' not in o.name ]
            self.operators_name  = [ o.name for o in ops]

            self.all_output_tensors = {tensor for op in ops for tensor in op.outputs}
            self.all_input_tensors  = {tensor for op in ops for tensor in op.inputs}
            self.all_output_names   = {tensor.name for op in ops for tensor in op.outputs}
            self.all_input_names    = {tensor.name for op in ops for tensor in op.inputs}
            self.tensors       = list(self.all_input_tensors | self.all_output_tensors)
            self.tensor_names  = [ t.name for t in self.tensors]
            for o in ops:
                    o.outputs_idx = [ self.tensor_names.index(t.name) for t in o.outputs ]
            for o in ops:
                    o.inputs_idx  = [ self.tensor_names.index(t.name) for t in o.inputs ]

            self.output_tensors   = list({ i for i in ( self.all_output_tensors - self.all_input_tensors ) if ':0' in i.name and '/' not in i.name })
            self.outputs_idx      = [ self.tensors.index(i) for i in self.output_tensors]
            self.outputs_name     = [ self.tensors[o].name for o in self.outputs_idx ]

            self.input_tensors    = []
            self.inputs_idx       = []
            self.inputs_name      = []

            self.all_op_names     = list({op.name for op in ops})
            self.only_l0_outputs  = list({ i for i in (self.all_output_names - self.all_input_names) if ':0' in i and '/' not in i })
            self.only_inputs      = list({ i for i in (self.all_input_names - self.all_output_names) })

            self.feedables        = {i.name for i in self.all_input_tensors if tf.get_default_graph().is_feedable(i) and '/' not in i.name}

            self.all_output_names = list(self.all_output_names)
            self.all_input_names  = list(self.all_input_names)

            self.reset_refs()
            self.operate_order_list     = []

    def reset_refs(self): self.operator_refs = [0]*len(self.operators)

    def refs(self, operator_idx):
        refs = self.operator_refs[operator_idx]
        self.operator_refs[operator_idx] += 1
        return refs

    def print_operator(self, operator_idx):
        opcode = self.operators[operator_idx].opcode_index
        o_obj  = self.operators[operator_idx]
        o_nick = self.operators[operator_idx].nick
        print("dest_tensor {} {} {:16s} <= operator {} {:3d}(code {:2d}) = src {}".format(
                o_obj.outputs,
                self.tensors[o_obj.outputs[0]].type,
                self.tensors[o_obj.outputs[0]].name,
                o_nick,
                operator_idx,
                opcode,
                o_obj.inputs
            )
        )

    #     Generators      Focus        Consumers
    #Tensor ---- ope ---+ Tensor +---- ope --- Tensor
    #List               | List   |             List
    #                   |        |
    #Tensor ____ ope ___|        |____ ope --- Tensor
    #List                                      List
    def generators(self, tensors):
        # find subgraph input operator index
        output_operators_idxes = []
        source_tensor_idxes    = []
        assert type(tensors) == list,"tensors(index) must be list"
        for ope_idx, ope in enumerate(self.operators):
            ope_outs = ope.outputs_idx
            assert type(ope_outs) == list,"ope_outs(index) must be list"
            if len(set(ope_outs+tensors)) != len(ope_outs+tensors):
                output_operators_idxes.append(ope_idx)
                source_tensor_idxes.append(ope.inputs_idx)
        return output_operators_idxes, source_tensor_idxes

    def walk_from(self, tensor_idx, verbose=True):
        operators, tensors = self.generators(tensor_idx)
        for o, t in zip(operators, tensors):
            if self.refs(o)>0:continue
            self.walk_from(t, verbose)
            self.operate_order_list.append(o)
            if verbose: self.print_operator(o)

    def allocate_graph(self, verbose=False):
        if verbose: print("Allocatng Graph ..")
        self.reset_refs()
        self.operate_order_list     = []
        self.walk_from(self.outputs_idx, verbose=verbose)
        if verbose: print("Allocatng Graph done.")
        def op_inputs(idx):return [i for i in self.operators[idx].inputs]
        for i in self.operate_order_list:
            inputs = op_inputs(i)
            if len(inputs) > 0:
                for j in inputs:
                    print("input tensors[{}]: {} => {}".format(i,j.name,self.operators_name[i]))
                self.input_tensors = inputs
                break

g = graph(detection_graph)
g.allocate_graph(verbose=False)
#for i in g.operate_order_list:
    #print(g.operators[i].name)
#print('only_l0_outputs  :{}'.format(only_l0_outputs))
#print('only_inputs      :{}'.format(only_inputs))
#print('feedable tensors :{}'.format(feedables))

