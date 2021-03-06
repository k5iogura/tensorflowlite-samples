import os,sys,re
import struct
import json
import numpy as np
from pdb import *
from nnop import DEPTHWISE_CONV_2D, MAX_POOL_2D, CONV_2D, RELUx

getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default
class operator_code():
    def __init__(self, code_json):
        self.json         = code_json
        self.builtin_code = code_json.get('builtin_code')
        self.custom_code  = code_json.get('custom_code')

    def view(self):
        print(self.builtin_code)

class operator():
    def __init__(self, operator_idx, operator_json, operator_codes, tensors):
        self.idx     = operator_idx
        self.json    = operator_json
        self.tensors = tensors
        self.inputs  = getordef(operator_json, 'inputs',  [])
        self.outputs = getordef(operator_json, 'outputs', [])
        self.opcode_index    = getordef(operator_json, 'opcode_index',    0)
        self.builtin_options = getordef(operator_json, 'builtin_options', None)

        self.name    = self.opcode_name = operator_codes[self.opcode_index].builtin_code
        self.nick    = re.sub('[_AIUEO0-9]','',self.name)[:5]
        self.padding = 0

    def unsupported(self):
        print(self.name+" IS NOT SUPPORTED",self.outputs,self.name,self.inputs)
        #sys.exit(-1)

    def fully_connected(self, x, W, b):
        # x : width  x height
        # W : height x width
        # b : height
        y = np.sum(x*W,axis=1)
        z = y+b
        return z

    def clipping(self, tensor_idx):
        tensor = self.tensors[tensor_idx[0]]
        if tensor.min is not None and tensor.max is not None:
            tensor.data = np.clip(tensor.data, tensor.min, tensor.max)
        elif tensor.min is not None:
            tensor.data = np.maximum(tensor.data, tensor.max)
        elif tensor.max is not None:
            tensor.data = np.minimum(tensor.data, tensor.min)

    def eval(self):
        name = self.name
        if   name == 'ADD':     # Untested yet
            r = self.tensors[0].data
            for i in self.inputs[1:]:
                assert self.tensors[0].shape == self.tensors[i].shape,"Unmatch {} {}".format(
                                                    r.shape, self.tensors[i].shape)
                r += self.tensors[i].data
            self.clipping(self.outputs)
            return r
        elif name == 'AVERAGE_POOL_2D':   self.unsupported()
        elif name == 'CONCATENATION':
            _axis_ = getordef(self.builtin_options,'axis',None)
            if _axis_ is None:self.view('Invalid conatenation axis',cont=False)
            temp_ = []
            for t in self.inputs:
                temp_.append(self.tensors[t].data.tolist())
            assert len(temp_) > 0, "Invalid concatenation list"
            r = self.tensors[self.outputs[0]].data = np.concatenate(temp_, axis = _axis_)
            return r
        elif name == 'CONV_2D':
            CONV_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
        elif name == 'DEPTHWISE_CONV_2D':
            DEPTHWISE_CONV_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
        elif name == 'EMBEDDING_LOOKUP':  self.unsupported()
        elif name == 'FULLY_CONNECTED':
            x = self.tensors[self.inputs[0]].data.reshape(-1)
            w = self.tensors[self.inputs[1]].data
            b = self.tensors[self.inputs[2]].data
            r = self.fully_connected(x,w,b)
            _activation_ = getordef(self.builtin_options, 'fused_activation_function', None)
            if _activation_ is not None:
                if   "RELU"  in _activation_: r = RELUx(r, 0)
                elif "RELU1" in _activation_: r = RELUx(r, 1)
                elif "RELU6" in _activation_: r = RELUx(r, 6)
                else: print(_activation_+' not supported')
            self.tensors[self.outputs[0]].data = r
            self.clipping(self.outputs)
            return r
        elif name == 'HASHTABLE_LOOKUP':  self.unsupported()
        elif name == 'L2_NORMALIZATION':  self.unsupported()
        elif name == 'L2_POOL_2D':        self.unsupported()
        elif name == 'LOCAL_RESPONSE_NORMALIZATION': self.unsupported()
        elif name == 'LOGISTIC':
            sigmoid = lambda x : 1 / (1 + np.exp(-x))
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = sigmoid(np.clip(x,-100,100))
            return r
        elif name == 'LSH_PROJECTION':    self.unsupported()
        elif name == 'LSTM':              self.unsupported()
        elif name == 'MAX_POOL_2D':
            MAX_POOL_2D(self, self.outputs, self.inputs)
            self.clipping(self.outputs)
        elif name == 'RELU':
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = RELUx(x, 0)
            return r
        elif name == 'RELU6':
            x = self.tensors[self.inputs[0]].data
            r = self.tensors[self.outputs[0]].data = RELUx(x, 6)
            return r
        elif name == 'RESHAPE':
            s = getordef(self.builtin_options, 'new_shape', None)
            x = self.tensors[self.inputs[0]].data
            if s is None: s = self.tensors[self.inputs[1]].data
            r = self.tensors[self.outputs[0]].data = x.reshape(tuple(s))
            return r
        elif name == 'RESIZE_BILINEAR':   self.unsupported()
        elif name == 'RNN':               self.unsupported()
        elif name == 'SOFTMAX':
            assert len(self.inputs) == 1, "SOFTMAX not support dim {}".format(self.inputs)
            beta = getordef(self.builtin_options, 'beta', 1.0)
            assert beta != 0, "SOFTMAX not support beta {}".format(beta)
            # x  = np.exp(self.tensors[self.inputs[0]].data - np.max(self.tensors[self.inputs[0]].data))
            input_tensor = self.tensors[self.inputs[0]]
            x  = np.exp(beta*(input_tensor.data - np.max(input_tensor.data)))
            r  = self.tensors[self.outputs[0]].data = x/np.sum(x)
            return r
        elif name == 'SPACE_TO_DEPTH':    self.unsupported()
        elif name == 'SVDF':              self.unsupported()
        elif name == 'TANH':              self.unsupported()
        elif name == 'CONCAT_EMBEDDINGS': self.unsupported()
        elif name == 'SKIP_GRAM':         self.unsupported()
        elif name == 'CALL':              self.unsupported()
        elif name == 'CUSTOM':            self.unsupported()

    def view(self, msg=None, cont=True):
        if msg is not None: print("\n***\n*** "+msg+"\n***")
        print("operator[{}]({}:{}) outputs {} inpus {}".format(self.idx, self.nick, self.opcode_index, self.outputs, self.inputs))
        print("  builtin_options : {} padding@run {}".format(self.builtin_options, self.padding))
        for o in self.outputs: self.tensors[o].view()
        for i in self.inputs:  self.tensors[i].view()
        assert cont,"Fatal Error occurrence at operator"

class tensor():
    def __init__(self, tensor_idx, tensor_json, buffers):
        self.idx    = tensor_idx
        self.json   = tensor_json
        self.shape  = getordef(tensor_json, 'shape', [])
        self.type   = getordef(tensor_json, 'type',  'FLOAT32')
        self.name   = getordef(tensor_json, 'name',  'nothing')
        self.buffer = getordef(tensor_json, 'buffer', None)

        if self.buffer is not None:
            self.buff = buffers[self.buffer].get('data')
            if self.buff is not None:
                self.buff = self.dataWtype(self.buff, self.type, self.shape)
                self.data = self.buff.copy()
            else:
                self.data = np.zeros(tuple(self.shape),dtype=self.type2np(self.type))
        else:
            self.buffer = -1

        self.quantization = getordef(tensor_json,'quantization',{})
        self.scale = self.max = self.min = self.zero_point = None
        if self.quantization != {}:
            self.scale      = getordef(self.quantization, 'scale', None)
            self.max        = getordef(self.quantization, 'max', None)
            self.min        = getordef(self.quantization, 'min', None)
            self.zero_point = getordef(self.quantization, 'zero_point', None)

            if self.scale is not None:
                assert len(self.scale) == 1,"Json format error len(scale)="+str(len(self.scale))
                self.scale = self.scale[0]
            elif self.max is not None or self.min is not None or self.zero_point is not None:
                self.scale = 1.0

            if self.max is not None:
                assert len(self.max) == 1,"Json format error len(max)="+str(len(self.max))
                self.max = self.max[0]

            if self.zero_point is not None:
                assert len(self.zero_point) == 1,"Json format error len(zero_point)="+str(len(self.zero_point))
                self.zero_point = self.zero_point[0]

            if self.min is not None:
                assert len(self.min) == 1,"Json format error len(min)="+str(len(self.min))
                self.data  = (self.scale * self.data + self.min).astype(np.float32)
                self.min   = self.min[0]

            elif self.zero_point is not None:
                self.min   =  self.scale * self.zero_point
                self.data  = (self.scale * (self.data.astype(np.int32) - self.zero_point)).astype(np.float32)

    def list2int(self, bdy, idx, Nbyte):
        val = 0
        for s, i in enumerate(range(idx, idx+Nbyte)): val += bdy[i]<<(8*s)
        return val

    def list2float(self, bdy, idx, Nbyte):
        val = self.list2int(bdy,idx,Nbyte)
        frm = "%0"+str(2*Nbyte)+"x"
        sp  = frm%val
        flt = struct.unpack('!f',sp.decode('hex'))[0]
        return flt

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

    def set(self, img):
        assert type(img) == np.ndarray,"Input image type must be numpy.ndarray but got "+str(type(img))
        assert img.dtype == self.type2np(self.type),"Cannot set tensor: expect {} but {}".format(self.type,img.dtype)
        if (self.max < img.max() or self.min > img.min()):
            print("Warnning: Suppoted float32 only so coverting input to float32")
            img = ( self.scale * img + self.min ).astype(np.float32)
        self.data = img
        return self.data

    def view(self, msg=None, cont=True):
        if msg is not None: print("\n***\n*** "+msg+"\n***")
        print("tensors[{}]({}) buffer:{}".format(self.idx, self.name, self.buffer))
        print("  type@json :{} type@run :{}".format(self.type,self.data.dtype))
        print("  shape@json:{} shape@run:{}".format(self.shape, self.data.shape))
        print("  quantization:{}".format(self.quantization))
        assert cont,"Fatal Error occurrence at tensor"

class graph:
    def __init__(self,json_text='detect.json'):
        with open(json_text) as j: root = json.load(j)

        # subgraph          = /subgraphs[0]
        # inputs            = /subgraphs[0]/inputs
        # outputs           = /subgraphs[0]/outputs
        # operators         = /subgraphs[0]/operators
        # tensors           = /subgraphs[0]/tensors
        self.subgraph_dict  = subgraph_dict = root['subgraphs'][0]
        self.inputs         = subgraph_dict['inputs']
        self.outputs        = subgraph_dict['outputs']

        self.tensors = []
        for idx, t in enumerate(subgraph_dict['tensors']):
            gtnsr = tensor(idx, t, root['buffers'])
            self.tensors.append(gtnsr)

        self.operator_codes = []
        for idx, o in enumerate(root['operator_codes']):
            oprtr_cd = operator_code(o)
            self.operator_codes.append(oprtr_cd)

        self.operators = []
        for idx, o in enumerate(subgraph_dict['operators']):
            oprtr = operator(idx, o, self.operator_codes, self.tensors)
            self.operators.append(oprtr)

        self.reset_refs()
        self.operate_order_list     = []

    def reset_refs(self): self.operator_refs = [0]*len(self.operators)

    def refs(self, operator_idx):
        refs = self.operator_refs[operator_idx]
        self.operator_refs[operator_idx] += 1
        return refs

    def generators(self, tensors):
        # find subgraph input operator index
        output_operators_idxes = []
        source_tensor_idxes    = []
        for ope_idx, ope in enumerate(self.operators):
            ope_outs = ope.outputs
            if len(set(ope_outs+tensors)) != len(ope_outs+tensors):
                output_operators_idxes.append(ope_idx)
                source_tensor_idxes.append(ope.inputs)
        return output_operators_idxes, source_tensor_idxes

    def consumers(self, tensors):
        # find subgraph output operator index
        input_operators_idxes = []
        distin_tensor_idxes   = []
        for ope_idx, ope in enumerate(self.operators):
            ope_ins = ope.inputs
            if len(set(ope_ins+tensors)) != len(ope_ins+tensors):
                input_operators_idxes.append(ope_idx)
                distin_tensor_idxes.append(ope.outputs)
        return input_operators_idxes, distin_tensor_idxes

    def print_operator(self, operator_idx):
        opcode = self.operators[operator_idx].opcode_index
        o_obj  = self.operators[operator_idx]
        o_nick = self.operators[operator_idx].nick
        print("dist_tensor {} <= operator {} {}(code {}) = src {} data_idx    {} <= {}".format(
                o_obj.outputs, o_nick, operator_idx, opcode, o_obj.inputs,
                [self.tensors[i].buffer for i in o_obj.outputs],
                [self.tensors[i].buffer for i in o_obj.inputs ]
            )
        )

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
        self.walk_from(self.outputs, verbose)
        for order, operator_idx in enumerate(self.operate_order_list):
            pass

    def invoke(self, verbose=True):
        if verbose: print("----- INVOKING      -----")
        for order, operator_idx in enumerate(self.operate_order_list):
            operator = self.operators[operator_idx]
            #for i in self.inputs:   # Check only
            #    input_ = self.tensors[i]
            #    assert tuple(input_.shape)==input_.data.shape,"Input shape mismatch {} {}".format(
            #            self.tensors[i].shape, self.tensors[i].data.shape)
            ans = operator.eval()
            if verbose: operator.view()
        if verbose: print("----- DONE --------------")
        return ans

if __name__ == '__main__':
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import argparse
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    args = argparse.ArgumentParser()
    def chF(f): return f if os.path.exists(f) else sys.exit(-1)
    args.add_argument('-j',"--json",       type=chF, default='detect.json')
    args = args.parse_args()

    g=graph(args.json)
    g.allocate_graph()
    questions=1
    corrects =0
    for i in range(questions):
        number_img, number_out = mnist.test.next_batch(1)
        g.tensors[g.inputs[0]].set(number_img)
        y = g.invoke(verbose=False)
        gt = np.argmax(number_out)
        pr = np.argmax(y)
        if gt!=pr:
            print("incorrenct:",gt,pr)
        else:
            corrects+=1
    print("accurracy %.3f %d/%d"%(1.0*corrects/questions,corrects,questions))

