import numpy as np
import sys
from pdb import *

#def CONV_2D(self, img, name, stride = 2, padding = 1, filter_size = 3):
#def CONV_2D(self, name, img):
def CONV_2D(operator, outputs, inputs, verbose=True):
    getordef = lambda json,key,default:json.get(key) if json.get(key) is not None else default

    stride           = getordef(operator.builtin_options, 'stride_h', 2)
    padding          = getordef(operator.builtin_options, 'padding',  0)
    depth_multiplier = getordef(operator.builtin_options, 'depth_multiplier',  1)
    activate         = getordef(operator.builtin_options, 'fused_activation_function', None)
    tensor_idx_input, tensor_idx_filter, tensor_idx_bias = inputs
    tensor_input     = operator.tensors[tensor_idx_input]
    tensor_filter    = operator.tensors[tensor_idx_filter]
    tensor_bias      = operator.tensors[tensor_idx_bias]
    filter_size      = tensor_filter.data.shape[1] # kernel height NHWC

    patches = []
    output_ = []
    shape = tensor_input.data.shape
    
    # Calculating output shape
    output_height = (tensor_input.shape[0] - filter_size + 2*padding) / stride + 1
    output_width  = (tensor_input.shape[1] - filter_size + 2*padding) / stride + 1
    
    # (32,32) -> (32,32,1)
    if tensor_input.data.ndim == 2: tensor_input.data = np.expand_dims(tensor_input.data, axis = 2)
    
    # Ensuring that output_height will be an integer
    if output_height - int(output_height) != 0:
        pad_vertical = (padding, 0)
    else:
        pad_vertical = (padding, padding)

    # Ensuring that output_width will be an integer
    if output_width - int(output_width) != 0:
        pad_horizontal = (padding, 0)
    else:
        pad_horizontal = (padding, padding)
    
    # Padding along height and width
    tensor_input.data = np.pad(tensor_input.data, (pad_horizontal, pad_vertical, (0,0)), mode='constant')
    
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
            patches.append(img[row_start:row_end, col_start:col_end, :]) ##M
    
    # Performing convolution with each patch, for every filter. (Technically, correlation).
    for filter_, bias in zip(F, B):
        temp_ = []
        for patch_idx, patch in enumerate(patches):
            temp_.append(np.sum(patch * filter_) + bias)
        output_.append(np.array(temp_).reshape(int(output_height), int(output_width)))
    
    output_ = np.transpose(np.array(output_), (1,2,0))
    
    # Printing model summary after initialization
    if verbose:
        print(name, '(input):', shape)
        print(name, '(output):', output_.shape)
        
    return output_

