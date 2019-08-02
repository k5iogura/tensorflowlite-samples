# mnist example

## Convert Chainer model to tflite

```python
$ python convert_mnist_model.py
```

`minist.tflite` file will be generated.

### How to compile

To build tensorflow-lite.a, follow [Using tensorflow-lite.a builds a.out](https://github.com/k5iogura/tensorflowlite-samples/blob/master/CNN_NumberDetector/05_TensorflowLite_CPP/README.md)  

### Build mnist 

```
 $ make.sh
```

build/mnist executable binary will be generated.  

### Prepare MNIST dataset

Download and extract MINST dataset to `data` directory.

```
$ mkdir data
$ cd data
$ ../download-minist.sh
$ gunzip *.gz
```

### Run

```
$ cd build
$ ./mnist (mnist.tflite) (mnist_data_directory)
```
