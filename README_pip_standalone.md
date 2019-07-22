# tensrflowlite standalone pip

To use tensorflowlite without full tensorflow version, use build_pip_package.sh.  
Below reproduction of build_pippackage.sh.  
build_pip_package.sh is supported master branch as of now.  

## Select base system to install tflite_runtime  

### A. Into docker with ubuntu 16.04 bash,  
```
 $ docker run -it --rm --net host -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority ubuntu:16.04 bash
```
### B. Into VirtualBox with ubuntu 16.04 bash,  

### Install tflite_runtime python module via below,  
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

### Check installation with tensorflowlite-samples repo.  

**Synopsis on python for tflite_runtime module**  

- from tflite_runtime import interpreter as tflr  
- interpreter = tflr.Interpreter(model_path="foo.tflite")  

```
 # pip install opencv-python
 # cd / ; git clone https://github.com/k5iogura/tensorflowlite-samples;cd tensorflowlite-samples
 # python dog_detector_tflite_runtime.py
   ...
   INFO: Initialized TensorFlow Lite runtime.
   ('location:', array([[[0.1904766 , 0.14921203, 0.7673927 , 0.76377976],
        [0.13206425, 0.60138065, 0.29834086, 0.89910287],
        [0.36077502, 0.1715548 , 0.9427432 , 0.4286785 ],
        [0.2241779 , 0.9015626 , 0.27170756, 0.93789315],
        [0.14713457, 0.09790109, 0.21337369, 0.12656319],
        [0.21864796, 0.83779806, 0.28488708, 0.9029984 ],
        [0.33766866, 0.2987341 , 0.7207978 , 0.7266861 ],
        [0.15676886, 0.08010703, 0.20985907, 0.10536472],
        [0.4080763 , 0.07923777, 0.6704359 , 0.27708873],
        [0.4061941 , 0.15655129, 0.6812936 , 0.3007803 ]]], dtype=float32))
   ('classes :', array([[ 1.,  2., 17.,  2.,  0.,  2.,  1.,  0., 61., 61.]], dtype=float32))
   ('score   :', array([[0.75      , 0.7109375 , 0.64453125, 0.5234375 , 0.51171875,
        0.44921875, 0.37890625, 0.37890625, 0.3671875 , 0.35546875]],
         dtype=float32))
   ('num dets:', array([10.], dtype=float32))
   1.846FPS
   0.750(114.000 109.000 586.000 442.000) 1
   0.711(461.000 76.000 690.000 171.000) 2
   0.645(131.000 207.000 329.000 543.000) 17
   
 # ls result.jpg
   result.jpg
 # apt install feh
   feh result.jpg
```

**July.22,2019**
