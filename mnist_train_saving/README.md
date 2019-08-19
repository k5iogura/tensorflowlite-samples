# mnist train and saving model example  
[Original GIST](https://gist.github.com/bigsnarfdude/95c19664f5f8aa5b8b403308c5d42b23)

This is training and inference from ground. Training process outputs ckpt directory and frozen protobuff file. You can infer MNIST task with ckpt directory or frozen pb file or tflite file.  
Need tflite_convert tool on tensorflow-v1.10.  

### Train  
```
 $ python mnist4ML.py
 $ ls ckpt *.pb
   mnist_frozen.pb
   ckpt:
   checkpoint  mnist.ckpt-100.data-00000-of-00001  mnist.ckpt-100.index  mnist.ckpt-100.meta
```

### Infer with ckpt directory.  
```
 $ python mnist_infe_ckpt.py
   精度
   0.9002
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
