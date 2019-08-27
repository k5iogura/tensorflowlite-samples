import numpy as np
import sys,os
import math
from pdb import *

# NHWC : input  tensor shape   = ( batch,  h, w, in_ch )
# NHWC : output tensor shape   = ( batch,  h, w, in_ch )
# 1HWC : filter tensor shape   = ( 1,      k, k, in_ch ) at DEPTHWISE
# CHWC : filter tensor shape   = ( out_ch, k, k, in_ch ) at CONV_2D
# C    : bias   tensor shape   = ( in_ch               )

def CONV_2D(operator, outputs, inputs, verbose=True):
    getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default

    stride           = getordef(operator.builtin_options, 'stride_h', 2)
    padding          = getordef(operator.builtin_options, 'padding',  0)
    depth_multiplier = getordef(operator.builtin_options, 'depth_multiplier',  0)
    activate         = getordef(operator.builtin_options, 'fused_activation_function', None)
    tensor_idx_input, tensor_idx_filter, tensor_idx_bias = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_idx_output= outputs[0]
    tensor_output    = operator.tensors[tensor_idx_output]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    shape = tensor_input.data.shape
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

    # Calculating output shape
    #output_height = (tensor_input.shape[0] - filter_size + 2*padding) / stride + 1
    #output_width  = (tensor_input.shape[1] - filter_size + 2*padding) / stride + 1
    
    # (32,32) -> (32,32,1)
    #if tensor_input.data.ndim == 2: tensor_input.data = np.expand_dims(tensor_input.data, axis = 2)
    
    # Ensuring that output_height will be an integer
    #if output_height - int(output_height) != 0:
    #    pad_vertical = (padding, 0)
    #else:
    #    pad_vertical = (padding, padding)

    # Ensuring that output_width will be an integer
    #if output_width - int(output_width) != 0:
    #    pad_horizontal = (padding, 0)
    #else:
    #    pad_horizontal = (padding, padding)

    # <by padding>
    _pad = int(math.ceil(((output_height - 1)*stride - tensor_input.shape[1] + filter_size)/2))
    # Padding along height and width
    #set_trace()
    #tensor_input.data = np.pad(tensor_input.data, (pad_horizontal, pad_vertical, (0,0)), mode='constant')
    tensor_input.data = np.pad(tensor_input.data, ((0,0),(_pad,_pad),(_pad,_pad),(0,0)), mode='constant')
    # output 1,28,28,32
    # input  1,34,34,32 <= changed
    # filter 1,5,5,32
    # bias   32
    
    # Appending current input for backpropagation
    #self.inputs[name] = tensor_input.data
    B = tensor_bias.data
    F = tensor_filter.data
    
    # Cutting the image into patches of size (filter_H, filter_W, input_channels)
    for row in range(int(output_height)):
        for col in range(int(output_width)):     
            row_start = row*stride
            row_end = row_start + filter_size
            col_start = col*stride
            col_end = col_start + filter_size
            #patches.append(tensor_input.data[:, row_start:row_end, col_start:col_end, :]) ##M
            # apatch 1,5,5,32
            apatch=tensor_input.data[:, row_start:row_end, col_start:col_end, :]
            assert apatch.shape == tensor_filter.data.shape,"Failed {} {}".format(apatch.shape, tensor_filter.data.shape)
            patches.append(apatch.tolist())
    # patches N,5,5,32
    patches = np.concatenate(patches, axis=0)
    # Performing convolution with each patch, for every filter. (Technically, correlation).
    temp_ = []  # for DepthSizeConv
    for filter_, bias in zip(F, B):
        # filter_ 5,5,32
        for patch_idx, patch_ in enumerate(patches):
            # patch_ 5,5,32
            #conv = (np.sum(patch_ * filter_) + bias)              # for CONV
            conv = (np.sum(patch_ * filter_, axis=(0,1)) + bias)   # for DepthSizeConv
            temp_.append(conv.tolist())
        # output_.append(np.array(temp_).reshape(int(output_height), int(output_width))) # for CONV
    #output_ = np.transpose(np.array(output_), (1,2,0)) # for CONV
    output_ = np.asarray(temp_).reshape((1, output_height, output_width, -1))
    assert output_.shape == tensor_output.data.shape
    # Printing model summary after initialization
    #if verbose:
    #    print(name, '(input):', shape)
    #    print(name, '(output):', output_.shape)
        
    return output_

