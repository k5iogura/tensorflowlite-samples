import numpy as np
import sys

try:
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
except:
    print('TensorFlow contrib library is used to load MNIST data')
    sys.exit()

try:
    from keras.utils import to_categorical
except:
    print('keras.utils is used to convert labels to one-hot representation.')
    sys.exit()


class network:
    
    """
    A convolutional neural network coded exclusively on NumPy.
    
    Args:
    self.lr -> Learning Rate
    self.std -> Standard deviation of normal initializer
    self.epochs -> Number of epochs
    self.params -> Dict that holds all the trainable parameters
    self.inputs -> Dict that holds the input vars for backprop
    self.names -> Dict that holds the names of operations to do backprop
    """
    
    def __init__(self, lr, epochs, std=0.03):
        self.params = {}
        self.inputs = {}
        self.names = []
        self.std = std
        self.lr = lr
        self.epochs = epochs
        
    def train(self):
        
        # Loads MNIST data
        self.load_data()
        
        # Training Loop
        for self.epoch in range(self.epochs):
            
            if self.epoch == 0:
                print('Model Summary (Epoch 0):')
            else:
                print('Epoch:', self.epoch)
                
            for self.i in range(55000):
                self.forward(self.data[self.i,:,:], (self.y[self.i,:]).reshape(1, -1))
                self.backward()
                print('Iteration:', self.i, 'Loss:', self.loss, end = '\r')
    
    def forward(self, input_, y):
        
        """
        Builds and executes the forward pass of a CNN.
        """
        
        x = self.conv(input_, name='conv1', filter_count=4, stride=2, padding=1)
        x = self.relu(x)
        x = self.dense(x, name='dense1', output_ch=10)
        self.loss = self.softmax_with_cross_entropy(x, y)
    
    def backward(self):
        
        """
        Executes backpropagation for the built CNN. Note that, you only need to build
        the network in the forward() function. You do not need to modify this function.
        """
        
        # A reverse trace of our "computational graph".
        trace = list(reversed(self.names))
        
        # In case multiple conv/dense operations are defined.
        seen = set()
        
        # Traverses the trace and does backpropagation along the way.
        for name in trace:
            type_ = name.split('_')[0]
            op = name.split('_')[1]
            
            if type_ == 'conv':
                if op not in seen:
                    seen.add(op)
                    self.conv_backward(self.inputs[op], self.delta, lr=self.lr)
            
            if type_ == 'dense':
                if op not in seen:
                    seen.add(op)
                    self.delta = self.dense_backward(self.inputs[op], self.delta, lr=self.lr)
            
            if type_ == 'loss':
                self.delta = self.softmax_with_cross_entropy_backward(self.inputs['loss'])
                
    def load_data(self):
        """
        MNIST dataloader. Here's the only non-numpy part
        Dataloader -> tensorflow.contrib.
        One-Hot encoder -> keras.utils
        """
        train, _, test = read_data_sets('.')
        self.data = np.array(train._images).reshape(55000,28,28)
        self.data = np.pad(self.data, ((0,0), (2,2), (2,2)), mode='constant')
        self.y = to_categorical(train._labels)
        
        self.test_data = np.array(test._images).reshape(10000,28,28)
        self.test_data = np.pad(self.test_data, ((0,0), (2,2), (2,2)), mode='constant')
        self.test_y = to_categorical(test._labels)
    
    def conv(self, img, filter_count, name, stride = 2, padding = 1, filter_size = 3):
        
        """
        Defines or executes the convolution operation. Automatically gets initialized during
        the first pass.
        """
        patches = []
        output_ = []
        shape = img.shape
        
        # Calculating output shape
        output_height = (img.shape[0] - filter_size + 2*padding) / stride + 1
        output_width = (img.shape[1] - filter_size + 2*padding) / stride + 1
        
        # (32,32) -> (32,32,1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis = 2)
        
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
        
        """
        If it's the first iteration, we need to define the convolution operation. The below
        code ensures that the operation is defined only in the first operation. It also fetches
        the updated weights from the previous iteration's backprop
        """
        
        if name + '_bias' not in self.params:
            B = np.zeros(filter_count)
            self.params[name + '_bias'] = B
            self.names.append('conv_' + name)
        else:
            B = self.params[name + '_bias']

        if name + '_filter' not in self.params:
            F = self.std * np.random.randn(filter_count, filter_size, filter_size, img.shape[2]) ##M
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
        if self.i == 0 and self.epoch == 0:
            print(name, '(input):', shape)
            print(name, '(output):', output_.shape)
            
        return output_
    
    def dense(self, input_, name, output_ch = 10):
        
        # Appending current input for backpropagation
        self.inputs[name] = input_
        
        # Ravelling input to feed it to a feed-forward network
        if input_.ndim == 3:
            input_ = input_.reshape(1, input_.shape[0]*input_.shape[1]*input_.shape[2])

        """
        If it's the first iteration, we need to define the dense operation. The below
        code ensures that the operation is defined only in the first operation. It also fetches
        the updated weights from the previous iteration's backprop
        """
        
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
        if self.i == 0 and self.epoch == 0:
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
    
    def dense_backward(self, input_, delta, lr=0.1):
        
        # Checks if input needs to be ravelled
        flag = False
        if input_.ndim == 3:
            flag = True
            shape = input_.shape
            input_ = input_.reshape(1, input_.shape[0]*input_.shape[1]*input_.shape[2])
        
        # Fetching current weights and bias
        weights = self.params['dense1_weight']
        bias = self.params['dense1_bias']
        
        # Calculating derivative to conv operation
        delta_to_input = np.matmul(weights, delta.T).T
        
        # Calculating updates using SGD
        weight_updates = lr * np.matmul(input_.T, delta)
        bias_updates = lr * delta.sum(axis = 0)
    
        # Updating weights and biases
        self.params['dense1_weight'] += weight_updates
        self.params['dense1_bias'] += bias_updates
        
        # Unravel derivative to send to conv layer
        if flag:
            delta_to_input = np.reshape(delta_to_input, (shape[0], shape[1], shape[2]))

        return delta_to_input

    def conv_backward(self, img, delta, lr=0.1):
        """
        Works on the principle that:
        weight_updates = Convolution(input, derivative_till_output)
        """

        patches = []
        weights_update = []
        
        # (32,32) -> (32,32,1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis = 2)
    
        # Fetching current weights and bias
        weights = self.params['conv1_filter']
        bias = self.params['conv1_bias']
        
        # Fetching filter sizes and target sizes
        output_height = weights.shape[1]
        output_width = weights.shape[2]
        filter_size = delta.shape[0]
        
        # padding = 0
        stride = int((img.shape[0] - delta.shape[0]) / (output_height - 1))
        
        # Getting input patches to convolve
        for row in range(int(output_height)):
            for col in range(int(output_width)):     
                row_start = row*stride
                row_end = row_start + filter_size
                col_start = col*stride
                col_end = col_start + filter_size
                patches.append(img[row_start:row_end, col_start:col_end, :]) ##M
        
        # The derivative_till_output (delta) is our filter. Making it compatible to convolve 
        F = np.transpose(delta, (2,0,1))
        if F.ndim == 3:
            F = np.expand_dims(F, axis=3)
        
        # Calculating the update
        for filter_ in F:
            temp_ = []
            for patch_idx, patch in enumerate(patches):
                temp_.append(np.sum(patch * filter_))
            weights_update.append(np.array(temp_).reshape(int(output_height), int(output_width), weights.shape[3]))
        
        # Updating weights and biases.
        weights_update = np.array(weights_update)
        self.params['conv1_filter'] += lr * weights_update
        self.params['conv1_bias'] += lr * np.sum(delta, axis=(0,1))

if __name__ == '__main__':

    net = network(lr = 0.001, epochs = 10)
    net.train()
