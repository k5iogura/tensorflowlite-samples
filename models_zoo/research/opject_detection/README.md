# Object Detection Demo

## how to prepare to run on python2  

```
 # pip install matplotlib==2.1.1  
 # pip install opencv-python
 $ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
 $ unzip protobuf.zip
 $ cp bin/protoc  $YOUR_COMMAND_PATH
```

## download model zoo  

```
 $ cd tensorflowlite-samples/tensorflow  
 $ git clone https://github.com/tensorflow/models
 $ cp ../models_zoo/research/opject_detection/object_detection_tutorial.py models/research/object_detection/
 $ cd models/research
 $ protoc object_detection/protos/*.proto --python_out=.
 $ cd object_detection
```

## how to run  

```
 $ python object_detection_tutorial.py
```

## Appendex  
[Use node_pb.py to check contents of graph file](./README_search_pbfile.md)  

**Oct.07, 2019**  

