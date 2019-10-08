# Use node_pb.py to check contencts of pbfile  

```
 $ python node_pb.py tensorflowlite-samples/mnist_cnn_json/mnist_frozen.pb
   PATH_TO_FROZEN_GRAPH:/home/hst20076433/t-samples/mnist_cnn_json/mnist_frozen.pb
   only_l0_outputs  :set([u'outputX:0'])
   only_inputs      :set([])
   feedable tensors :set([u'Variable_2:0', u'MatMul:0', u'Relu_2:0', u'Variable_3:0',
       u'Reshape_1:0', u'MaxPool_1:0', u'Reshape:0', u'add_1:0', u'Relu_1:0', u'inputX:0',
       u'MaxPool:0', u'Variable:0', u'add_3:0', u'MatMul_1:0', u'add:0', u'Conv2D:0',
       u'Relu:0', u'add_2:0', u'Variable_7:0', u'Conv2D_1:0', u'Variable_5:0',
       u'Variable_4:0', u'Variable_1:0', u'Variable_6:0'])
```

**feedable tensors** denotes attribute as is_feedable(), so one of these may be input of graph.  

