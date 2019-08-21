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

### Infer with ckpt directory.  
```
 $ python mnist_infe_ckpt.py
   精度
   0.9002
 $ ls *.pb
   mnist_frozen.pb
```

**Aug.20,2019**  

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

### Infe with json file by python using numpy only  
```
 $ flatc --strict-json -t ../schema_v3.fbs -- mnist.tflite
 $ ls *.json
   mnist.json
 $ python tfjson.py -j mnist.json
   dist_tensor [2] <= operator 0(code 0) = src [3, 1, 0] data_idx    [3] <= [5, 1, 4]
   dist_tensor [4] <= operator 1(code 1) = src [2] data_idx    [2] <= [3] 
   ('incorrenct:', 9, 3)
   ('incorrenct:', 5, 3)
   ('incorrenct:', 5, 9)
   ('incorrenct:', 7, 9)
   ('incorrenct:', 9, 3)
   ('incorrenct:', 7, 4)
   ('incorrenct:', 8, 5)
   accurracy 0.930 93/100
```
