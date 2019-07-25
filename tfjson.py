import os,sys
import json
from pdb import *

class walker:
    def __init__(self):
        with open('detect.json') as j: root = json.load(j)

        # datas_list   = /buffers/data
        cleanup_lambda = lambda i: i.get('data') if i.get('data') is not None else []
        self.datas_list     = [cleanup_lambda(i) for i in root['buffers']]

        # subgraph          = /subgraphs[0]
        subgraph_dict       = root['subgraphs'][0]
        # inputs            = /subgraphs[0]/inputs
        # outputs           = /subgraphs[0]/outputs
        # operators         = /subgraphs[0]/operators
        # tensors           = /subgraphs[0]/tensors
        self.inputs_list    = subgraph_dict['inputs']
        self.outputs_list   = subgraph_dict['outputs']
        self.operators_list = subgraph_dict['operators']
        self.tensors_list   = subgraph_dict['tensors']

        # operator_codes = /operator_codes
        self.operator_codes_list = root['operator_codes']

        self.reset_flag()

    def reset_flag(self): self.operators_flag = [0]*len(self.operators_list)

    def flag(self, operator_idx):
        flag = self.operators_flag[operator_idx]
        self.operators_flag[operator_idx] += 1
        return flag

    def generators(self, tensors):
        # find subgraph input operator index
        output_operators_idxes = []
        source_tensor_idxes    = []
        for ope_idx, ope in enumerate(self.operators_list):
            ope_outs = ope['outputs']
            if len(set(ope_outs+tensors)) != len(ope_outs+tensors):
                output_operators_idxes.append(ope_idx)
                source_tensor_idxes.append(ope['inputs'])
        return output_operators_idxes, source_tensor_idxes

    def consumers(self, tensors):
        # find subgraph output operator index
        input_operators_idxes = []
        distin_tensor_idxes   = []
        for ope_idx, ope in enumerate(self.operators_list):
            ope_ins = ope['inputs']
            if len(set(ope_ins+tensors)) != len(ope_ins+tensors):
                input_operators_idxes.append(ope_idx)
                distin_tensor_idxes.append(ope['outputs'])
        return input_operators_idxes, distin_tensor_idxes

#     Generators      Focus        Consumers
#Tensor ---- ope ---+ Tensor +---- ope --- Tensor
#List               | List   |             List
#                   |        |
#Tensor ____ ope ___|        |____ ope --- Tensor
#List                                      List
w=walker()
def walk(tensor_idx):
    operators, tensors = w.generators(tensor_idx)
    for o, t in zip(operators,tensors):
        if w.flag(o)>0:continue
        walk(t)
        o_obj = w.operators_list[o]
        print("dist_tensor {} <= operator {} = src_tensor {}".format( o_obj['outputs'], o, o_obj['inputs']) )

walk(w.outputs_list)

