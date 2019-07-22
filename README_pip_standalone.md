# tensrflowlite standalone pip

To use tensorflowlite without full tensorflow version, use build_pip_package.sh.  
Below reproduction of build_pippackage.sh.  
build_pip_package.sh is supported master branch as of now.  

Into docker with ubuntu.16.04 bash,  
```
 $ docker run -it --rm --net host -e DISPLAY=$DISPLAY ubuntu:16.04 bash
```

Install tflite_runtime python module via below,  
```
 # apt install git curl vim vim-syntax-gtk ctags make cmake
 # apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy python3 python
 # apt install -y python-pip python3-pip libopencv-dev
 # apt install python-numpy
 # pip install numpy

 # git clone https://github.com/tensorflow/tensorflow
 # cd tensorflow
 # ./tensorflow/lite/tools/pip_package/build_pip_package.sh
 # find /tmp/ -iname \*.whl
  /tmp/tflite_pip/python/dist/tflite_runtime-1.14.0-cp27-cp27mu-linux_x86_64.whl
 # pip install /tmp/tflite_pip/python/dist/tflite_runtime-1.14.0-cp27-cp27mu-linux_x86_64.whl
```

Check installation,  
```
# python
>>> from tflite_runtime import interpreter as tflr
>>> interpreter = tflr.Interpreter(model_path="foo.tflite")
```

**July.22,2019**
