# mnist train and saving model example  
[Original GIST](https://gist.github.com/bigsnarfdude/95c19664f5f8aa5b8b403308c5d42b23)

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
 $ python mnist_infe_tflite.py
   精度
   0.9002
```
