import os,sys
import json
import numpy as np
from pdb import *

class graph:
    def __init__(self,json_text='detect.json'):
        with open(json_text) as j: root = json.load(j)

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

    def print_operator(self, operator_idx):
        opcode= self.get_opcode_index(operator_idx)
        o_obj = self.operators_list[operator_idx]
        print("dist_tensor {} <= operator {}(code {}) = src_tensor {}".format( o_obj['outputs'], operator_idx, opcode, o_obj['inputs']) )

    def get_opcode_index(self,operator):
        opcode_index = self.operators_list[operator].get('opcode_index')
        return opcode_index if opcode_index is not None else 0

    def list2int(self, bdy, idx, Nbyte):
        val = 0
        for s, i in enumerate(range(idx, idx+Nbyte)): val += bdy[i]<<(8*s)
        return val

    def list2float(self, bdy, idx, Nbyte):
        return np.float32(self.list2int(bdy,idx,Nbyte))

    def get_tensor(self, tensor_idx):
        idx = self.tensors_list[tensor_idx].get('buffer')
        shp = self.tensors_list[tensor_idx].get('shape')
        typ = self.tensors_list[tensor_idx].get('type')
        bdy = self.datas_list[idx]
        if   typ == 'FLOAT32':
            if len(bdy)==0: return np.asarray([], np.float32)
            return np.asarray( [self.list2float(bdy, i, 4) for i in range(0,len(bdy),4)], np.float32 ).reshape(shp)
        elif typ == 'FLOAT16':
            if len(bdy)==0: return np.asarray([], np.float16)
            return np.asarray( [self.list2float(bdy, i, 2) for i in range(0,len(bdy),2)], np.float16 ).reshape(shp)
        elif typ == 'INT32':
            if len(bdy)==0: return np.asarray([], np.int32)
            return np.asarray( [self.list2int(bdy, i, 4) for i in range(0,len(bdy),4)], np.int32 ).reshape(shp)
        elif typ == 'UINT8':
            if len(bdy)==0: return np.asarray([], np.uint8)
            return np.asarray(bdy, np.uint8).reshape(shp)
        elif typ == 'INT64':
            if len(bdy)==0: return np.asarray([], np.int64)
            return np.asarray( [self.list2int(bdy, i, 8) for i in range(0,len(bdy),8)], np.int64 ).reshape(shp)

    #     Generators      Focus        Consumers
    #Tensor ---- ope ---+ Tensor +---- ope --- Tensor
    #List               | List   |             List
    #                   |        |
    #Tensor ____ ope ___|        |____ ope --- Tensor
    #List                                      List
    def walk_from(self, tensor_idx, proc=None, verbose=True):
        operators, tensors = self.generators(tensor_idx)
        for o, t in zip(operators, tensors):
            if self.flag(o)>0:continue
            self.walk_from(t, proc, verbose)
            if verbose: self.print_operator(o)

            if proc is not None:
                opcode_index = self.get_opcode_index(o)
                src_numpy_list = []
                src_tensors_list = self.operators_list[o].get('inputs')
                for t in src_tensors_list: src_numpy_list.append(self.get_tensor(t))

                dst_numpy_list = []
                dst_tensors_list = self.operators_list[o].get('outputs')
                for t in dst_tensors_list: dst_numpy_list.append(self.get_tensor(t))
            #    proc(dst_numpy_list, opcode_index, src_numpy_list)

def exec_operation(np_dst, opcode_index, np_src):
    for i in np_dst:
        assert i is not None,np_src
    print(
        [i.shape for i in np_dst],
        opcode_index,
        [j.shape for j in np_src]
    )

g=graph('detect.json')
g.walk_from(g.outputs_list, exec_operation, verbose=True)

