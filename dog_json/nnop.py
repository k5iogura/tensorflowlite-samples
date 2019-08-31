import numpy as np
import sys,os
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from pdb import *

# NHWC : input  tensor shape   = ( batch,  h, w, in_ch )
# NHWC : output tensor shape   = ( batch,  h, w, in_ch )
# 1HWC : filter tensor shape   = ( 1,      k, k, in_ch ) at DEPTHWISE
# CHWC : filter tensor shape   = ( out_ch, k, k, in_ch ) at CONV_2D
# C    : bias   tensor shape   = ( in_ch               )

def RELUx(numpy_in, val=0, leaky=None):
    assert numpy_in.dtype != np.uint8,"RELU not supports {}".format(np.uint8)
    numpy_out = numpy_in.copy()
    if val > 1:                    # RELUx
        numpy_out[numpy_out < 0]   = 0
        numpy_out[numpy_out > val] = val
    elif val == 1:                 # RELU1
        numpy_out[numpy_out < -1]  = -1
        numpy_out[numpy_out > val] =  1
    elif leaky is not None:        # LEAKY RELU
        numpy_out[numpy_out < 0]  *= leaky
    else:                          # RELU
        numpy_out[numpy_out < 0]   = 0
    return numpy_out

def CONV_2D(operator, outputs, inputs, verbose=True):
    getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default

    stride           = getordef(operator.builtin_options, 'stride_h', 2)
    padding          = getordef(operator.builtin_options, 'padding',  0)
    _activation_     = getordef(operator.builtin_options, 'fused_activation_function', None)
    tensor_idx_input, tensor_idx_filter, tensor_idx_bias = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    
    # stride 1
    # output 1,14,14,64
    # input  1,14,14,32
    # filter 64,5,5,32
    # bias   64

    # <by padding>
    _pad = ((output_height - 1)*stride - input_shape[1] + filter_size)/2.
    _pad = Decimal(_pad).quantize(Decimal(0), rounding=ROUND_HALF_UP)
    _pad = int(_pad)
    operator.padding = _pad
    #_pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    # Padding along height and width
    if _pad >= 0:
        tensor_input.data = np.pad(
            tensor_input.data,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    else:
        operator.view("Invalid padding size",cont=False)
    # output 1,14,14,64
    # input  1,14,14,32
    # filter 64,5,5,32
    # bias   64
    
    B = tensor_bias.data
    F = tensor_filter.data
    
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            #patches.append(tensor_input.data[:, row_start:row_end, col_start:col_end, :]) ##M
            # apatch 1,5,5,32
            apatch=tensor_input.data[:, row_start:row_end, col_start:col_end, :]
            if apatch.shape[1:]!=tensor_filter.data.shape[1:]:
                set_trace()
            assert apatch.shape[1:]==tensor_filter.data.shape[1:],"Failed {} {}".format(
                                                apatch.shape, tensor_filter.data.shape)
            patches.append(apatch.tolist())
    # patches 14*14,5,5,32
    patches = np.concatenate(patches, axis=0)
    # temp_ = []  # for DepthWiseConv
    for filter_, bias in zip(F, B):
        temp_ = []  # for CONV
        # filter_ 5,5,32
        for patch_idx, patch_ in enumerate(patches):
            # patch_ 5,5,32
            conv = (np.sum(patch_ * filter_) + bias)              # for CONV as scaler
            #conv = (np.sum(patch_ * filter_, axis=(0,1)) + bias)   # for DepthWiseConv
            temp_.append(conv)
            #temp_.append(conv.tolist())                            # for DepthWiseConv
        #temp_ 14*14
        output_.append(np.array(temp_).reshape(output_height, output_width)) # for CONV
    # output_ 14,14,64
    output_ = np.transpose(np.array(output_), (1,2,0)) # for CONV
    # output_ 1,14,14,64
    output_ = output_[np.newaxis, :]
    #output_ = np.asarray(temp_).reshape((1, output_height, output_width, -1)) # for DepthWiseConv
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape,"Mismatch {} {}".format(
                            output_.shape,tensor_output.data.shape)
    tensor_output.data = output_

    return output_

def DEPTHWISE_CONV_2D(operator, outputs, inputs, verbose=True):
    getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default

    stride           = getordef(operator.builtin_options, 'stride_h', 2)
    padding          = getordef(operator.builtin_options, 'padding',  0)
    depth_multiplier = getordef(operator.builtin_options, 'depth_multiplier',  0)
    _activation_     = getordef(operator.builtin_options, 'fused_activation_function', None)
    tensor_idx_input, tensor_idx_filter, tensor_idx_bias = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    
    # <by depth_multiplier>
    # output 1,28,28,32
    # input  1,28,28,1  (depth_multiplier==32)
    # filter 1,5,5,32
    # bias   32
    if depth_multiplier>0:
        np_concat = []
        for m in range(depth_multiplier):
            np_concat.append(tensor_input.data)
        tensor_input.data = np.concatenate(np_concat,axis=3)
    # output 1,28,28,32
    # input  1,28,28,32 <= changed
    # filter 1,5,5,32
    # bias   32

    # <by padding>
    _pad = ((output_height - 1)*stride - input_shape[1] + filter_size)/2.
    _pad = Decimal(_pad).quantize(Decimal(0), rounding=ROUND_HALF_UP)
    _pad = int(_pad)
    operator.padding = _pad
    #_pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    # Padding along height and width
    if _pad != 0:
        tensor_input.data = np.pad(
            tensor_input.data,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    else:
        operator.view(cont=False)
    # output 1,28,28,32
    # input  1,34,34,32 <= changed
    # filter 1,5,5,32
    # bias   32
    
    B = tensor_bias.data
    F = tensor_filter.data
    
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            #patches.append(tensor_input.data[:, row_start:row_end, col_start:col_end, :]) ##M
            # apatch 1,5,5,32
            apatch=tensor_input.data[:, row_start:row_end, col_start:col_end, :]
            assert apatch.shape == tensor_filter.data.shape,"Failed {} {}".format(
                                                apatch.shape, tensor_filter.data.shape)
            patches.append(apatch.tolist())
    # patches N,5,5,32
    patches = np.concatenate(patches, axis=0)
    temp_ = []  # for DepthWiseConv
    for filter_, bias in zip(F, B):
        # temp_ = []  # for CONV
        # filter_ 5,5,32
        for patch_idx, patch_ in enumerate(patches):
            # patch_ 5,5,32
            #conv = (np.sum(patch_ * filter_) + bias)              # for CONV
            conv = (np.sum(patch_ * filter_, axis=(0,1)) + bias)   # for DepthWiseConv
            temp_.append(conv.tolist())
        # output_.append(np.array(temp_).reshape(int(output_height), int(output_width))) # for CONV
    #output_ = np.transpose(np.array(output_), (1,2,0)) # for CONV
    output_ = np.asarray(temp_).reshape((1, output_height, output_width, -1))
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape,"Mismatch {} {}".format(
                            output_.shape,tensor_output.data.shape)
    tensor_output.data = output_

    return output_

def MAX_POOL_2D(operator, outputs, inputs, verbose=True):
    getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default

    stride           = getordef(operator.builtin_options, 'stride_h', 2)
    padding          = getordef(operator.builtin_options, 'padding',  0)
    filter_size      = getordef(operator.builtin_options, 'filter_width',  0)
    filter_size      = getordef(operator.builtin_options, 'filter_height',  filter_size )
    _activation_     = getordef(operator.builtin_options, 'fused_activation_function', None)
    tensor_idx_input = inputs[0]
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]

    patches = []
    output_ = []
    input_shape = tensor_input.data.shape
    output_height, output_width = tensor_output.data.shape[1:3]
    
    # input  1,28,28,32
    # output 1,14,14,32

    # <by padding>
    _pad = int(math.ceil(((output_height - 1)*stride - input_shape[1] + filter_size)/2))
    operator.padding = _pad
    # Padding along height and width
    if _pad != 0:
        tensor_input.data = np.pad(
            tensor_input.data,
            ((0,0),(_pad,_pad),(_pad,_pad),(0,0)),
            mode='constant', constant_values=(0,0)
        )
    else:
        operator.view(cont=False)
    # input  1,28,28,32
    # output 1,14,14,32
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            # apatch N,2,2,32
            apatch=tensor_input.data[:, row_start:row_end, col_start:col_end, :]
            mpatch=np.max(apatch, axis=(1), keepdims=True)   # N,1,2,32
            mpatch=np.max(mpatch, axis=(2), keepdims=False)  # N,1,32
            # mpatch N,14*14,32
            patches.append(mpatch.tolist())
    # patches N,14*14,32
    patches = np.concatenate(patches, axis=1)
    # patches N,14,14,32
    patches = patches.reshape(-1, output_height, output_width, patches.shape[-1])
    output_ = patches
    if _activation_ is not None:
        if   "RELU"  in _activation_: output_ = RELUx(output_, 0)
        elif "RELU1" in _activation_: output_ = RELUx(output_, 1)
        elif "RELU6" in _activation_: output_ = RELUx(output_, 6)
        else: print(_activation_+' not supported')
    assert output_.shape == tensor_output.data.shape
    tensor_output.data = output_

    return output_

