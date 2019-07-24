import os,sys
import json

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

w=walker()
ten=[w.inputs_list]
while True:
    ope, ten = w.consumers(ten[0])
    if len(ope)<=0:break
    code = w.operators_list[ope[0]].get('opcode_index')
    if code is None:code=0
    syno = "{} {} {} {}".format(ope,code,w.operators_list[ope[0]]['inputs'],ten)
    print(syno)
    #print(ope,operators_list[ope[0]]['builtin_options_type'],operators_list[ope[0]]['inputs'],ten)

