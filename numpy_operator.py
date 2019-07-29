import numpy as np
import sys
from pdb import *

class nn_operator:
    
    """
    Args:
    self.std -> Standard deviation of normal initializer
    self.params -> Dict that holds all the trainable parameters
    self.inputs -> Dict that holds the input vars for backprop
    self.names -> Dict that holds the names of operations to do backprop
    """
    
    def __init__(self, std=0.03, verbose=True):
        self.params  = {}
        self.inputs  = {}
        self.names   = []
        self.std     = std
        self.options = {}
        self.verbose = verbose

    def add_CONV_2D(self, name, np_filter, np_bias, options):
        self.params[name + '_options'] = options
        self.params[name + '_bias']    = np_bias.copy()
        self.params[name + '_filter']  = np_filter.copy()

    #def CONV_2D(self, img, name, stride = 2, padding = 1, filter_size = 3):
    def CONV_2D(self, name, img):
        assert self.params.get(name+'_filter') is not None, name

        stride      = self.options.get('stride_h') if self.options.get('stride_h') is not None else 2
        padding     = self.options.get('padding')  if self.options.get('padding')  is not None else 1
        activate    = self.options.get('fused_activation_function')
        filter_size = self.params[name+'_filter'].shape[1] # kernel height
        if filter_size is None: filter_size = 3

        patches = []
        output_ = []
        shape = img.shape
        
        # Calculating output shape
        output_height = (img.shape[0] - filter_size + 2*padding) / stride + 1
        output_width = (img.shape[1] - filter_size + 2*padding) / stride + 1
        
        # (32,32) -> (32,32,1)
        if img.ndim == 2: img = np.expand_dims(img, axis = 2)
        
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
        img = np.pad(img, (pad_horizontal, pad_vertical, (0,0)), mode='constant')
        
        # Appending current input for backpropagation
        self.inputs[name] = img
        
        if name + '_bias' not in self.params:
            B = np.zeros(filter_count)
            self.params[name + '_bias'] = B # _bias : ( in_ch )
            self.names.append('conv_' + name)
        else:
            B = self.params[name + '_bias']

        if name + '_filter' not in self.params:
            F = self.std * np.random.randn(filter_count, filter_size, filter_size, img.shape[2]) # _filter : ( go_ch, ksize, ksize, in_ch )
            self.params[name + '_filter'] = F
        else:
            F = self.params[name + '_filter']
        
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
        if self.verbose:
            print(name, '(input):', shape)
            print(name, '(output):', output_.shape)
            
        return output_
    
    def dense(self, input_, name, output_ch = 10):
        
        # Appending current input for backpropagation
        self.inputs[name] = input_
        
        # Ravelling input to feed it to a feed-forward network
        if input_.ndim == 3:
            input_ = input_.reshape(1, input_.shape[0]*input_.shape[1]*input_.shape[2])

        if name + '_bias' not in self.params:
            bias = np.zeros(output_ch)
            self.params[name + '_bias'] = bias    
            self.names.append('dense_' + name)
        else:
            bias = self.params[name + '_bias']

        if name + '_weight' not in self.params:
            weight = self.std * np.random.randn(input_.shape[1], output_ch)
            self.params[name + '_weight'] = weight
        else:
            weight = self.params[name + '_weight']
        
        # Compute the matrix multiplication
        output_ = np.matmul(input_, weight) + bias
        
        # Printing model summary after initialization
        if self.verbose:
            print(name, '(ravel):', input_.shape)
            print(name, '(output):', output_.shape)
        
        return output_
    
    def relu(self, input_):
        # A very efficient relu code 
        return np.where(input_ > 0, input_, np.zeros_like(input_))
    
    def softmax(self, x):
        # -np.max(x) normalizes the exponents
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def softmax_with_cross_entropy(self,x,y):
        
        # EPS prevents log(0) condition
        EPS = 1e-12
        p = self.softmax(x)
        
        # Adding softmax operation in the first iteration
        if 'loss' not in self.params:
            self.names.append('loss_')
        
        # Fetching targets and loss at every iteration 
        self.inputs['loss'] = y
        self.params['loss'] = p
        
        # Calculating output
        log_likelihood = - y * np.log(p + EPS)
        loss = np.sum(log_likelihood) / y.shape[0]
        return loss
    
    def softmax_with_cross_entropy_backward(self, y):
        # Derivative of loss wrt softmax with cross entropy
        delta = self.params['loss'] - y
        return delta
    
