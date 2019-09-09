# mnist train and saving CNN model example  
[Reference Page](https://qiita.com/haminiku/items/36982ae65a770565458d)

This is training and inference from ground. Training process outputs ckpt directory and frozen protobuff file. You can infer MNIST task with ckpt directory or frozen pb file or tflite file.  
Need tflite_convert tool on tensorflow-v1.10.  

### Train  
```
 $ python mnist4ML.py
 $ ls ckpt
   ckpt:
   checkpoint  mnist.ckpt-100.data-00000-of-00001  mnist.ckpt-100.index  mnist.ckpt-100.meta
```

### Infer and create frozen pb file without Dropout layer with ckpt directory.  
```
 $ python mnist_infe_ckpt.py
   精度
   0.9002
 $ ls *.pb
   mnist_frozen.pb
```

### Infer with frozen pb.  
```
 $ python mnist_infe_pb.py
   精度
   0.9002
```

### Infer with tflite by tensorflow-lite.  
```
 $ tflite_convert --graph_def_file mnist_frozen.pb --input_arrays=inputX --output_arrays=outputX  --output_file=mnist.tflite
 $ ls *.tflite
   mnist.tflite
 $ python mnist_infe_tflite.py
   incorrect truth:prediction = 2 : 3
   incorrect truth:prediction = 9 : 4
   incorrect truth:prediction = 8 : 7
   incorrect truth:prediction = 2 : 1
   incorrect truth:prediction = 3 : 8
   精度
   0.9002
```

### Infer with tflite by python(fbapi.py) using numpy only  

Step.1 to make tflite flatbuffers api.  
```
 $ flatc --python ../schema_v3.fbs
   or
 $ ./make_tflite.sh
 $ ls tflite/*.py
tflite/ActivationFunctionType.py   tflite/__init__.py                           tflite/QuantizationParameters.py
tflite/AddOptions.py               tflite/L2NormOptions.py                      tflite/ReshapeOptions.py
tflite/Buffer.py                   tflite/LocalResponseNormalizationOptions.py  tflite/ResizeBilinearOptions.py
tflite/BuiltinOperator.py          tflite/LSHProjectionOptions.py               tflite/RNNOptions.py
tflite/BuiltinOptions.py           tflite/LSHProjectionType.py                  tflite/SkipGramOptions.py
tflite/CallOptions.py              tflite/LSTMOptions.py                        tflite/SoftmaxOptions.py
tflite/ConcatEmbeddingsOptions.py  tflite/Model.py                              tflite/SpaceToDepthOptions.py
tflite/ConcatenationOptions.py     tflite/OperatorCode.py                       tflite/SubGraph.py
tflite/Conv2DOptions.py            tflite/Operator.py                           tflite/SVDFOptions.py
tflite/DepthwiseConv2DOptions.py   tflite/Padding.py                            tflite/Tensor.py
tflite/FullyConnectedOptions.py    tflite/Pool2DOptions.py                      tflite/TensorType.py
```

Step.2 infer mnist with tflite  
```
 $ python fbapi.py
...
Creating Graph done.
Allocatng Graph ..
dist_tensor [9] <= operator RSHP 0(code 4) = src [16, 10] data_idx    [15] <= [10, 16]
dist_tensor [6] <= operator DPTHW 1(code 1) = src [9, 11, 1] data_idx    [14] <= [15, 18, 12]
dist_tensor [4] <= operator MXPLD 2(code 3) = src [6] data_idx    [17] <= [14]
dist_tensor [7] <= operator CNVD 3(code 0) = src [4, 12, 0] data_idx    [7] <= [17, 5, 4]
dist_tensor [5] <= operator MXPLD 4(code 3) = src [7] data_idx    [6] <= [7]
dist_tensor [8] <= operator FLLYC 5(code 2) = src [5, 13, 3] data_idx    [11] <= [6, 3, 1]
dist_tensor [15] <= operator FLLYC 6(code 2) = src [8, 14, 2] data_idx    [8] <= [11, 2, 9]
dist_tensor [17] <= operator SFTMX 7(code 5) = src [15] data_idx    [13] <= [8]
Allocatng Graph done.
('incorrenct:', 6, 4)
('incorrenct:', 4, 6)
('incorrenct:', 1, 3)
('incorrenct:', 9, 7)
('incorrenct:', 2, 7)
('incorrenct:', 7, 9)
accurracy 0.940 94/100
```

### Infer with json file by python using numpy only  
```
 $ flatc --strict-json -t ../schema_v3.fbs -- mnist.tflite
 $ ls *.json
   mnist.json
 $ python tfjson.py -j mnist.json
   dist_tensor [9]  <= operator RSHP  0(code 4) = src [16, 10]   data_idx    [15] <= [10, 16]
   dist_tensor [6]  <= operator DPTHW 1(code 1) = src [9, 11, 1] data_idx    [14] <= [15, 18, 12]
   dist_tensor [4]  <= operator MXPLD 2(code 3) = src [6]        data_idx    [17] <= [14]
   dist_tensor [7]  <= operator CNVD  3(code 0) = src [4, 12, 0] data_idx    [7]  <= [17, 5, 4]
   dist_tensor [5]  <= operator MXPLD 4(code 3) = src [7]        data_idx    [6]  <= [7]
   dist_tensor [8]  <= operator FLLYC 5(code 2) = src [5, 13, 3] data_idx    [11] <= [6, 3, 1]
   dist_tensor [15] <= operator FLLYC 6(code 2) = src [8, 14, 2] data_idx    [8]  <= [11, 2, 9]
   dist_tensor [17] <= operator SFTMX 7(code 5) = src [15]       data_idx    [13] <= [8]
```

Included operators in mnist.tflite.  
```
  "operator_codes": [
    { "builtin_code": "CONV_2D" },
    { "builtin_code": "DEPTHWISE_CONV_2D" },
    { "builtin_code": "FULLY_CONNECTED" },
    { "builtin_code": "MAX_POOL_2D" },
    { "builtin_code": "RESHAPE" },
    { "builtin_code": "SOFTMAX" }
  ]
```
As of 30.Aug,2019, supported operators are **CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED**.  

### For debug of own implemented operator see [TensorFlow Lite Interpreter get_tensor() #23384](../README_get_tensor.md)  

**Aug.30, 2019**  
