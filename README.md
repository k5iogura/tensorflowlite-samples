# tensorflowlite-samples

Reference  
[初心者に優しくないTensorflow Lite の公式サンプル](https://qiita.com/yohachi/items/434f0da356161e82c242)  
[Object Detection](https://www.tensorflow.org/lite/models/object_detection/overview)  

## tensorflowlite  
On CentOS7.5 install python3, tensorflow-gpu and opencv.  
python3  
```
 # yum install -y https://centos7.iuscommunity.org/ius-release.rpm
 # yum install -y python35u python35u-libs python35u-devel python35u-pip
 # cd /usr/bin;ln -s python3.5m python3
```

tensorflow and opencv  
```
 # python3 -m pip install tensorflow-gpu  
 # python3 -m pip install opencv-python  
 
 $ python3 -c "import tensorflow"
 $
 $ python3 -c "import cv2"
 $
```

## Download tflite and labels  

Goto [Object Detection](https://www.tensorflow.org/lite/models/object_detection/overview) and find sample model zip. As of now zip is coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip.  
```
 $ wget http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
 $ unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
 $ ls
   coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip detect.tflite labelmap.txt
```
- detect.tflite : includes weights and network  
- labelmap.txt  : includes detectable object labels  

## Run inference  

```
 $ python3 sample.py
 $
   INFO: Initialized TensorFlow Lite runtime.
   7.386FPS
```

