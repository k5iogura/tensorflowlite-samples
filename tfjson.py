import os,sys
import json
import numpy as np
from pdb import *

class graph:
    def __init__(self,json_text='detect.json'):
        with open(json_text) as j: root = json.load(j)

        self.invoke_layer = 1000

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

        self.datas_npy_list = [[]]*len(self.datas_list)
        self.datas_shp_list = [[]]*len(self.datas_list)
        self.datas_typ_list = [None]*len(self.datas_list)
        self.datas_qnt_list = [[]]*len(self.datas_list)
        for idx in range(len(self.tensors_list)):
            data_idx = self.tensors_list[idx].get('buffer')
            data_shp = self.tensors_list[idx].get('shape')
            data_typ = self.tensors_list[idx].get('type')
            data_qnt = self.tensors_list[idx].get('quantization')
            self.datas_shp_list[data_idx] = data_shp
            self.datas_npy_list[data_idx] = self.get_tensor(idx)
            self.datas_typ_list[data_idx] = data_typ
            self.datas_qnt_list[data_idx] = data_qnt

        # operator_codes = /operator_codes
        self.operator_codes_list = root['operator_codes']

        self.reset_refs()
        self.operate_order_list     = []

    def reset_refs(self): self.operator_refs = [0]*len(self.operators_list)

    def refs(self, operator_idx):
        refs = self.operator_refs[operator_idx]
        self.operator_refs[operator_idx] += 1
        return refs

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
        print("dist_tensor {} <= operator {}(code {}) = src {} :data_idx {} <= {}".format(
                o_obj['outputs'], operator_idx, opcode, o_obj['inputs'],
                [self.tensors_list[i].get('buffer') for i in o_obj['outputs']],
                [self.tensors_list[i].get('buffer') for i in o_obj['inputs']]
            )
        )

    def get_tensor_quantization(self,tensor_idx):
        return self.tensors_list[tensor_idx].get('quantization')

    def get_builtin_code(self,operator_idx):
        opcode_index = self.get_opcode_index(operator_idx)
        builtin_code = self.operator_codes_list[opcode_index].get('builtin_code')
        return builtin_code if builtin_code is not None else 'Undefind'

    def get_opcode_index(self,operator_idx):
        opcode_index = self.operators_list[operator_idx].get('opcode_index')
        return opcode_index if opcode_index is not None else 0

    def list2int(self, bdy, idx, Nbyte):
        val = 0
        for s, i in enumerate(range(idx, idx+Nbyte)): val += bdy[i]<<(8*s)
        return val

    def list2float(self, bdy, idx, Nbyte):
        return np.float32(self.list2int(bdy,idx,Nbyte))

    def get_tensor_npy(self, tensor_idx):
        idx = self.tensors_list[tensor_idx].get('buffer')
        return self.datas_npy_list[idx]

    def get_tensor(self, tensor_idx):
        idx = self.tensors_list[tensor_idx].get('buffer')
        shp = self.tensors_list[tensor_idx].get('shape')
        typ = self.tensors_list[tensor_idx].get('type')
        bdy = self.datas_list[idx]
        if   typ == 'FLOAT32':
            if len(bdy)==0: return np.zeros(shp, np.float32)
            return np.asarray( [self.list2float(bdy, i, 4) for i in range(0,len(bdy),4)], np.float32 ).reshape(shp)
        elif typ == 'FLOAT16':
            if len(bdy)==0: return np.zeros(shp, np.float16)
            return np.asarray( [self.list2float(bdy, i, 2) for i in range(0,len(bdy),2)], np.float16 ).reshape(shp)
        elif typ == 'INT32':
            if len(bdy)==0: return np.zeros(shp, np.int32)
            return np.asarray( [self.list2int(bdy, i, 4) for i in range(0,len(bdy),4)], np.int32 ).reshape(shp)
        elif typ == 'UINT8':
            if len(bdy)==0: return np.zeros(shp, np.uint8)
            return np.asarray(bdy, np.uint8).reshape(shp)
        elif typ == 'INT64':
            if len(bdy)==0: return np.zeros(shp, np.int64)
            return np.asarray( [self.list2int(bdy, i, 8) for i in range(0,len(bdy),8)], np.int64 ).reshape(shp)
        elif typ == None and shp is not None:
            return np.zeros(shp, np.uint8).reshape(shp)
        assert False, "tensor_idx={} shape:{} type:{} bdy:{}".format(tensor_idx, shp, typ, bdy)

    #     Generators      Focus        Consumers
    #Tensor ---- ope ---+ Tensor +---- ope --- Tensor
    #List               | List   |             List
    #                   |        |
    #Tensor ____ ope ___|        |____ ope --- Tensor
    #List                                      List
    def walk_from(self, tensor_idx, func=None, verbose=True):
        operators, tensors = self.generators(tensor_idx)
        for o, t in zip(operators, tensors):
            if self.refs(o)>0:continue
            self.walk_from(t, func, verbose)
            self.operate_order_list.append(o)
            if verbose: self.print_operator(o)

            if func is not None:
                opcode_index = self.get_opcode_index(o)
                src_numpy_list = []
                src_tensors_list = self.operators_list[o].get('inputs')
                for t in src_tensors_list: src_numpy_list.append(self.get_tensor(t))

                dst_numpy_list = []
                dst_tensors_list = self.operators_list[o].get('outputs')
                for t in dst_tensors_list: dst_numpy_list.append(self.get_tensor(t))
                func(dst_numpy_list, opcode_index, src_numpy_list)

    def allocate_graph(self, verbose=True):
        self.walk_from(self.outputs_list, None, verbose)

    def invoke(self, verbose=True):
        if verbose: print("----- INVOKING      -----")
        for order, operator_idx in enumerate(self.operate_order_list):
            operator   = self.operators_list[operator_idx]
            src_tensor = operator.get('inputs')
            dst_tensor = operator.get('outputs')
            src_tensors_npy = [self.get_tensor_npy(tensor) for tensor in src_tensor]
            dst_tensors_npy = [self.get_tensor_npy(tensor) for tensor in dst_tensor]
            print("-----\ndst_tensor {} <= operator_idx {} <= src {}".format(dst_tensor, operator_idx, src_tensor))
            for tensor_idx in dst_tensor+src_tensor:
                qnt = self.get_tensor_quantization(tensor_idx)
                print("  tensor_idx {}:{}".format("%4d"%tensor_idx, qnt))
            exec_operation(self, dst_tensors_npy, operator_idx, src_tensors_npy)
            if order==self.invoke_layer:break
        if verbose: print("----- INVOKING DONE -----")

def exec_operation(graph, np_dst, operator_idx, np_src):
    opcode_index = graph.get_opcode_index(operator_idx)
    operator     = graph.operators_list[operator_idx]
    builtin_name = graph.get_builtin_code(operator_idx),
    print(
        "  dst_shape:{} <= {} <= src_shape:{}".format(
        [i.shape for i in np_dst],
        builtin_name,
        [j.shape for j in np_src]
    ))
#    set_trace()

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-j',"--json",       type=chF, default='detect.json')
    args.add_argument('-i',"--invoke_layer", type=int, default=0)
    args = args.parse_args()

    g=graph(args.json)
    g.invoke_layer = args.invoke_layer
    g.allocate_graph()
    g.invoke()

