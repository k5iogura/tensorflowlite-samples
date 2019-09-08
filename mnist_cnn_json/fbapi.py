# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
from   pdb import set_trace
from   inspect import getmembers

import struct

import tflite
from tflite.Model import Model
import tflite.BuiltinOptions
import tflite.TensorType

import tflite.AddOptions
import tflite.CallOptions
import tflite.ConcatenationOptions
import tflite.Conv2DOptions
import tflite.DepthwiseConv2DOptions
import tflite.FullyConnectedOptions
import tflite.L2NormOptions
import tflite.Pool2DOptions
import tflite.QuantizationParameters
import tflite.RNNOptions
import tflite.ReshapeOptions
import tflite.ResizeBilinearOptions
import tflite.SoftmaxOptions

import tflite.OperatorCode
import tflite.BuiltinOperator

def read_tflite_model(file):
    buf = open(file, "rb").read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    return model

def TensorType2String(TensorType):
    if TensorType == tflite.TensorType.TensorType.FLOAT32:
        return "FLOAT32"
    elif TensorType == tflite.TensorType.TensorType.FLOAT16:
        return "FLOAT16"
    elif TensorType == tflite.TensorType.TensorType.INT32:
        return "INT32"
    elif TensorType == tflite.TensorType.TensorType.INT8:
        return "INT8"
    elif TensorType == tflite.TensorType.TensorType.UINT8:
        return "UINT8"
    elif TensorType == tflite.TensorType.TensorType.INT64:
        return "INT64"
    elif TensorType == tflite.TensorType.TensorType.STRING:
        return "STRING"
    else:
        assert False,"Unknown:TensorType2String(TensorType)"+str(TensorType)

class tensor():
    def __init__(self, tensor_idx, tensor_fb, buffers_fb):
        self.idx    = tensor_idx
        self.shape  = list(tensor_fb.ShapeAsNumpy())
        self.type   = TensorType2String(tensor_fb.Type())
        self.name   = tensor_fb.Name()
        self.buffer = tensor_fb.Buffer()

        if buffers_fb[self.buffer].DataLength() > 0:
            self.buff = buffers_fb[self.buffer].DataAsNumpy()
            if self.buff is not None:
                self.buff = self.dataWtype(self.buff, self.type, self.shape)
                self.data = self.buff.copy()
            else:
                self.data = np.zeros(tuple(self.shape),dtype=self.type2np(self.type))
        else:
            self.buffer = -1

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
        assert type(bdy) == np.ndarray,"tensor:{} {}".format(self.idx, type(bdy))
        if   type_string=='FLOAT32': data = np.asarray([self.list2float(bdy, i, 4) for i in range(0,len(bdy),4)], np_type)
        elif type_string=='FLOAT16': data = np.asarray([self.list2float(bdy, i, 2) for i in range(0,len(bdy),2)], np_type)
        elif type_string=='INT32':   data = np.asarray([self.list2int(  bdy, i, 4) for i in range(0,len(bdy),4)], np_type)
        elif type_string=='INT64':   data = np.asarray([self.list2int(  bdy, i, 8) for i in range(0,len(bdy),8)], np_type)
        elif type_string=='UINT8':   data = np.asarray(                 bdy,                                      np_type)
        else : assert True, "Unsupported type"+type_string
        return data.reshape(tuple(shp))

class graph:
    def __init__(self,tflite='mnist.tflite'):
        self.model    = read_tflite_model(tflite)
        self.subgraph = self.model.Subgraphs(0)
        self.inputs   = list(self.subgraph.InputsAsNumpy())
        self.outputs  = list(self.subgraph.OutputsAsNumpy())
        buffers_fb    = [ self.model.Buffers(b) for b in range(self.model.BuffersLength()) ]
        self.tensors  = []
        for idx in range(self.subgraph.TensorsLength()):
            tensor_fb = self.subgraph.Tensors(idx)
            gtnsr = tensor(idx, tensor_fb, buffers_fb)
            self.tensors.append(gtnsr)

g = graph()

set_trace()
model = read_tflite_model('mnist.tflite')
sgl = model.SubgraphsLength()
if sgl!=0:
    sg = model.Subgraphs(0)
tsl = sg.TensorsLength()
print("tensors_length",tsl)

ts = sg.Tensors(0)

bfl = model.BuffersLength()
print("buffers length",bfl)

print("inputs",sg.InputsAsNumpy())
print("outputs",sg.OutputsAsNumpy())

opl = sg.OperatorsLength()
print("operators length",opl)

for idx in range(opl):
    op = sg.Operators(idx)
    inputs = op.InputsAsNumpy()
    outputs = op.OutputsAsNumpy()
    opcode_index = op.OpcodeIndex()
    if op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.Conv2DOptions:
        opt = tflite.Conv2DOptions.Conv2DOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        padding = opt.Padding()
        strideh = opt.StrideH()
        stridew = opt.StrideW()
        _activation_ = opt.FusedActivationFunction()
        print("Conv2DOptions",idx)
        print(padding, stridew, strideh, _activation_)

    elif op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions:
        opt = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        padding = opt.Padding()
        strideh = opt.StrideH()
        stridew = opt.StrideW()
        _activation_ = opt.FusedActivationFunction()
        depthmultiplier = opt.DepthMultiplier()
        print("DepthwiseConv2DOptions",idx)
        print(padding, stridew, strideh, _activation_,depthmultiplier)

    elif op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions:
        opt = tflite.FullyConnectedOptions.FullyConnectedOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        _activation_ = opt.FusedActivationFunction()
        print("FullyConnectedOptions",idx)
        print(_activation_)

    elif op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.SoftmaxOptions:
        opt = tflite.SoftmaxOptions.SoftmaxOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        beta = opt.Beta()
        print("SoftmaxOptions",idx)
        print(beta)

    elif op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions:
        opt = tflite.ReshapeOptions.ReshapeOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        newshape = list(opt.NewShapeAsNumpy())
        print("ReshapeOptions",idx)
        print(newshape)

    elif op.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions:
        opt = tflite.Pool2DOptions.Pool2DOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        padding = opt.Padding()
        strideh = opt.StrideH()
        stridew = opt.StrideW()
        filterwidth = opt.FilterWidth()
        filterheight = opt.FilterHeight()
        _activation_ = opt.FusedActivationFunction()
        print("Pool2DOptions",idx)
        print(padding, stridew, strideh, _activation_,filterwidth, filterheight)
    else:
        assert False,"Unknown:BuiltinOptions:"+str(op.BuiltinOptionsType())

s=TensorType2String(ts.Type())

def BuiltinCode2String(opcode_index):
    builtin_code = model.OperatorCodes(opcode_index).BuiltinCode()
    custom_code  = model.OperatorCodes(opcode_index).CustomCode()
    print("operator code {} builtin_code/custom_code = {}/{}".format(opcode_index,builtin_code,custom_code))
    if builtin_code == tflite.BuiltinOperator.BuiltinOperator.CONCATENATION:
        return "CONCATENATION"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.CONV_2D:
        return "CONV_2D"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D:
        return "DEPTHWISE_CONV_2D"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED:
        return "FULLY_CONNECTED"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.LOGISTIC:
        return "LOGISTIC"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.MAX_POOL_2D:
        return "MAX_POOL_2D"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RELU:
        return "RELU"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RELU6:
        return "RELU6"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.RESHAPE:
        return "RESHAPE"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.SOFTMAX:
        return "SOFTMAX"
    elif builtin_code == tflite.BuiltinOperator.BuiltinOperator.CUSTOM:
        return "CUSTOM"
    else:
        assert False,"Unknown "

ocl = model.OperatorCodesLength()
print("operator_code_length:",ocl)
for idx in range(ocl):
    name = BuiltinCode2String(idx)
    print("operator_code {} name : {}".format(idx,name))

