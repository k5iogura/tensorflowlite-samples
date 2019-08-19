import os,sys
import json
import numpy as np
from pdb import *

import numpy_operator
net = numpy_operator.nn_operator()


class operator_code():
    def __init__(self, code_json):
        self.json         = code_json
        self.builtin_code = code_json.get('builtin_code')
        self.custom_code  = code_json.get('custom_code')

    def view(self):
        print(self.builtin_code)

class operator():
    def __init__(self, operator_idx, operator_json, operator_codes):
        self.idx     = operator_idx
        self.json    = operator_json
        self.inputs  = operator_json.get('inputs')  if operator_json.get('inputs')  is not None else []
        self.outputs = operator_json.get('outputs') if operator_json.get('outputs') is not None else []
        self.opcode_index    = operator_json.get('opcode_index') if operator_json.get('opcode_index') is not None else 0
        self.builtin_options = operator_json.get('builtin_options')
        name = self.opcode_name = operator_codes[self.opcode_index].builtin_code
        return
        if   name == 'ADD': sys.exit(-1)
        elif name == 'AVERAGE_POOL_2D': sys.exit(-1)
        elif name == 'CONCATENATION': sys.exit(-1)
        elif name == 'CONV_2D': sys.exit(-1)
        elif name == 'DEPTHWISE_CONV_2D': sys.exit(-1)
        elif name == 'EMBEDDING_LOOKUP': sys.exit(-1)
        elif name == 'FULLY_CONNECTED': sys.exit(-1)
        elif name == 'HASHTABLE_LOOKUP': sys.exit(-1)
        elif name == 'L2_NORMALIZATION': sys.exit(-1)
        elif name == 'L2_POOL_2D': sys.exit(-1)
        elif name == 'LOCAL_RESPONSE_NORMALIZATION': sys.exit(-1)
        elif name == 'LOGISTIC': sys.exit(-1)
        elif name == 'LSH_PROJECTION': sys.exit(-1)
        elif name == 'LSTM': sys.exit(-1)
        elif name == 'MAX_POOL_2D': sys.exit(-1)
        elif name == 'RELU': sys.exit(-1)
        elif name == 'RELU6': sys.exit(-1)
        elif name == 'RESHAPE': sys.exit(-1)
        elif name == 'RESIZE_BILINEAR': sys.exit(-1)
        elif name == 'RNN': sys.exit(-1)
        elif name == 'SOFTMAX': sys.exit(-1)
        elif name == 'SPACE_TO_DEPTH': sys.exit(-1)
        elif name == 'SVDF': sys.exit(-1)
        elif name == 'TANH': sys.exit(-1)
        elif name == 'CONCAT_EMBEDDINGS': sys.exit(-1)
        elif name == 'SKIP_GRAM': sys.exit(-1)
        elif name == 'CALL': sys.exit(-1)
        elif name == 'CUSTOM': sys.exit(-1)

    def view(self):
        print(self.idx, self.inputs, self.outputs, self.opcode_index, self.opcode_name)

class tensor():
    def __init__(self, tensor_idx, tensor_json, buffers):
        self.idx    = tensor_idx
        self.json   = tensor_json
        self.shape  = tensor_json.get('shape')  if tensor_json.get('shape')  is not None else []
        self.type   = tensor_json.get('type')   if tensor_json.get('type')   is not None else 'FLOAT32'
        self.name   = tensor_json.get('name')   if tensor_json.get('name')   is not None else 'nothing'
        self.buffer = tensor_json.get('buffer')

        if tensor_json.get('quantization') is not None:
            quantization = self.quantization = tensor_json.get('quantization')
            qmax = self.max = quantization.get('max') if quantization.get('max') is not None else 0.
            qmin = self.min = quantization.get('min') if quantization.get('min') is not None else 0.
            scale= self.scale      = quantization.get('scale')      if quantization.get('scale') is not None else 0.
            zero = self.zero_point = quantization.get('zero_point') if quantization.get('zero_point') is not None else 0
        else:
            self.quantization = {}

        if self.buffer is not None:
            data = buffers[self.buffer].get('data')
            if data is not None:
                self.data = self.dataWtype(data, self.type, self.shape)
            else:
                self.data = np.zeros(tuple(self.shape),dtype=self.type2np(self.type))
        else:
            self.buffer = -1

    def list2int(self, bdy, idx, Nbyte):
        val = 0
        for s, i in enumerate(range(idx, idx+Nbyte)): val += bdy[i]<<(8*s)
        return val

    def list2float(self, bdy, idx, Nbyte):
        return np.float32(self.list2int(bdy,idx,Nbyte))

    def type2np(self,type_string):
        if type_string == 'FLOAT32': return np.float32
        if type_string == 'FLOAT16': return np.float16
        if type_string == 'INT32': return np.int32
        if type_string == 'INT64': return np.int64
        if type_string == 'UINT8': return np.uint8
        return np.float

    def dataWtype(self, bdy, type_string, shp):
        np_type = self.type2np(type_string)
        if   type_string=='FLOAT32': data = np.asarray([self.list2float(bdy, i, 4) for i in range(0,len(bdy),4)], np_type)
        elif type_string=='FLOAT16': data = np.asarray([self.list2float(bdy, i, 2) for i in range(0,len(bdy),2)], np_type)
        elif type_string=='INT32':   data = np.asarray([self.list2int(  bdy, i, 4) for i in range(0,len(bdy),4)], np_type)
        elif type_string=='INT64':   data = np.asarray([self.list2int(  bdy, i, 8) for i in range(0,len(bdy),8)], np_type)
        elif type_string=='UINT8':   data = np.asarray(                 bdy,                                      np_type)
        else : assert True, "Unsupported type"+type_string
        return data.reshape(tuple(shp))


    def view(self):
        print(self.json)
        print(self.idx, self.data)

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

        self.tensors_npy_list = [[]]*len(self.tensors_list)
        self.tensors_f32_list = [[]]*len(self.tensors_list)
        self.tensors_shp_list = [[]]*len(self.tensors_list)
        self.tensors_typ_list = [[]]*len(self.tensors_list)
        self.tensors_qnt_list = [[]]*len(self.tensors_list)
        self.datas_npy_list = [[]]*len(self.datas_list)
        self.datas_f32_list = [[]]*len(self.datas_list)
        self.datas_shp_list = [[]]*len(self.datas_list)
        self.datas_typ_list = [None]*len(self.datas_list)
        self.datas_qnt_list = [[]]*len(self.datas_list)
        for idx in range(len(self.tensors_list)):
            data_idx = self.tensors_list[idx].get('buffer')
            data_shp = self.tensors_list[idx].get('shape')
            data_typ = self.tensors_list[idx].get('type')
            data_qnt = self.tensors_list[idx].get('quantization')
            self.tensors_npy_list[idx] = self.datas_npy_list[data_idx] = self.get_tensor(idx)
            self.tensors_f32_list[idx] = self.datas_f32_list[data_idx] = self.get_tensor_npy_f32(idx)
            self.tensors_shp_list[idx] = self.datas_shp_list[data_idx] = data_shp
            self.tensors_typ_list[idx] = self.datas_typ_list[data_idx] = data_typ
            self.tensors_qnt_list[idx] = self.datas_qnt_list[data_idx] = data_qnt
        self.operator_codes_list = root['operator_codes']

        self.tensors = []
        for idx in range(len(self.tensors_list)):
            gtnsr = tensor(idx, subgraph_dict['tensors'][idx], root['buffers'])
            self.tensors.append(gtnsr)
            gtnsr.view()

        self.operator_codes = []
        for idx in range(len(self.operator_codes_list)):
            oprtr_cd = operator_code(root['operator_codes'][idx])
            self.operator_codes.append(oprtr_cd)

        self.operators = []
        for idx in range(len(subgraph_dict['operators'])):
            oprtr = operator(idx, subgraph_dict['operators'][idx], self.operator_codes)
            self.operators.append(oprtr)
            oprtr.view()

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
        print("dist_tensor {} <= operator {}(code {}) = src {} data_idx    {} <= {}".format(
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

    def get_tensor_npy_f32(self, tensor_idx):
        npy_tensor = self.get_tensor_npy(tensor_idx)
        qnt        = self.get_tensor_quantization(tensor_idx)
        qnt_min    = qnt.get('min')   if qnt.get('min') is not None else 0.0
        qnt_scale  = qnt.get('scale') if qnt.get('scale') is not None else 1.0
        qnt_zero   = qnt.get('zero_point')
        if npy_tensor.dtype == np.uint8:
            ui8_tensor = npy_tensor.copy().astype(np.int32)
            f32_tensor = qnt_scale * ui8_tensor.astype(np.float32) + qnt_min
        if npy_tensor.dtype == np.int32:
            f32_tensor = qnt_scale * npy_tensor
        return f32_tensor.astype(np.float32)

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
    def walk_from(self, tensor_idx, verbose=True):
        operators, tensors = self.generators(tensor_idx)
        for o, t in zip(operators, tensors):
            if self.refs(o)>0:continue
            self.walk_from(t, verbose)
            self.operate_order_list.append(o)
            if verbose: self.print_operator(o)

    def allocate_graph(self, verbose=True):
        self.walk_from(self.outputs_list, verbose)
        for order, operator_idx in enumerate(self.operate_order_list):
            pass

    def invoke(self, verbose=True):
        if verbose: print("----- INVOKING      -----")
        for order, operator_idx in enumerate(self.operate_order_list):
            operator   = self.operators_list[operator_idx]
            src_tensor = operator.get('inputs')
            dst_tensor = operator.get('outputs')
            src_tensors_npy = [self.tensors_f32_list[tensor] for tensor in src_tensor]
            dst_tensors_npy = [self.tensors_f32_list[tensor] for tensor in dst_tensor]
            builtin_name = self.get_builtin_code(operator_idx),
            print("-----\ndst_tensor {} <= operator_idx {} <= src {}".format(dst_tensor, operator_idx, src_tensor))
            print(
                "operator {} {} {} {}".format(
                operator_idx,
                builtin_name,
                operator.get('builtin_options_type'),
                operator.get('builtin_options')
            ))
            for tensor_idx in dst_tensor+src_tensor:
                qnt = self.get_tensor_quantization(tensor_idx)
                print("  tensor_idx {}:{}".format("%4d"%tensor_idx, qnt))
            print(
                "  dst_shape:{} <= {} <= src_shape:{}".format(
                [i.shape for i in dst_tensors_npy],
                builtin_name,
                [j.shape for j in src_tensors_npy]
            ))
            x = exec_operation(self, dst_tensors_npy, operator_idx, src_tensors_npy)
            if order==self.invoke_layer:break
        if verbose: print("----- INVOKING DONE -----")
        return x, src_tensors_npy, dst_tensors_npy

def exec_operation(graph, np_dst, operator_idx, np_src):
    opcode_index = graph.get_opcode_index(operator_idx)
    operator     = graph.operators_list[operator_idx]
    net.add_CONV_2D('conv1', np_src[1], np_src[2], operator.get('builtin_options'))
    return net.CONV_2D('conv1', np.zeros((300,300,3),np.float32))

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
    #x, src, dst = g.invoke()


