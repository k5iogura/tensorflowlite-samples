# SSD: Single Shot MultiBox Detector in TensorFlow

*This is modified [Original SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow/) site to reproduce.*  
*The Explanation for SSD VGG network is spellouted very well on same author's anothor site [here](https://github.com/balancap/SDC-Vehicle-Detection). It is very helpfull.*  
------------------

SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). Present TF checkpoints have been directly converted from SSD Caffe models.

The organisation is inspired by the TF-Slim models repository containing the implementation of popular architectures (ResNet, Inception and VGG). Hence, it is separated in three main parts:
* datasets: interface to popular datasets (Pascal VOC, COCO, ...) and scripts to convert the former to TF-Records;
* networks: definition of SSD networks, and common encoding and decoding methods (we refer to the paper on this precise topic);
* pre-processing: pre-processing and data augmentation routines, inspired by original VGG and Inception implementations.

## Requirement to reproduce  
- python(2.7.12)  
- tensorflow(1.13.1) [Community version](https://github.com/k5iogura/docker_docker/blob/master/README_tensorflow.md)  
- matplotlib(2.1.1)  
- opencv2(4.1.1)  
- jupyter(1.0.0 with python3)  

## SSD minimal example to infer only  

The [SSD Notebook](notebooks/ssd_notebook.ipynb) and its python version [ssd_notebook.py](notebook/ssd_notebook.py) contains a minimal example of the SSD TensorFlow pipeline. Shortly, the detection is made of two main steps: running the SSD network on the image and post-processing the output using common algorithms (top-k filtering and Non-Maximum Suppression algorithm).

Here are two examples of successful detection outputs:
![](../SSD-Tensorflow_files/ex1.png "SSD anchors")
![](../SSD-Tensorflow_files/ex2.png "SSD anchors")

### inference via pretrained ckpt files  
To run python scripts you first have to unzip the checkpoint files in ./checkpoint
```bash
  $ cd SSD-Tensorflow2
  $ unzip checkpoints/ssd_300_vgg.ckpt.zip -d checkpoints
```

and then run python to inference object without jupyter    
```
  $ cd notebook
  $ PYTHONPATH=../ python ssd_notebook.py
```

if you want to know how to make ssd_notebook.py from ssd_notebook.ipynb, following below,  

```
  $ cd notenbooks
  $ jupyter nbconvert --to python ssd_notebook.ipynb 
    [NbConvertApp] Converting notebook ssd_notebook.ipynb to python
    [NbConvertApp] Writing 4183 bytes to ssd_notebook.py

  $ vi ssd_notebook.py
  // comment out a get_ipython line 
  // #get_ipython().run_line_magic('matplotlib', 'inline')
  $ touch __init__.py
  $ PYTHONPATH=../ python ssd_notebook.py
```  
![](../SSD-Tensorflow_files/dog_result.jpg)  

### inference via protocol buffer file made by ssd_notebook.py.  
ssd_notebook.py generates also *frozen protobuf 'ssd_net_frozen.pb'*  
```
 $ ls *.pb
   ssd_net_frozen.pb
```

infer objects with ssd_net_frozen.pb.  
```
 $ python object_detection_pb.py
```

### inference via tflite(flatbuffers) made by tflite_convert command.  
To infer with tflite file that has float type of input and inference.  
```
  $ tflite_convert \
--graph_def_file=ssd_net_frozen.pb \
--output_file=./detect.tflite \
--output_format=TFLITE \
--input_arrays=Placeholder \
--input_shapes=300,300,3 \
--inference_input_type FLOAT \
--inference_type=FLOAT \
--mean_values=128 \
--std_dev_values=128 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--output_arrays="ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax/Reshape_1,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/softmax_5/Reshape_1" \
--allow_custom_ops

 $ ls *.tflite
   detect.tflite

 $ PYTHONPATH=.. python object_detection_tfl.py
```
![](../SSD-Tensorflow_files/dog_result_no_nms.jpg)  

Each class has a random color and without nms(Non Maximum Suppression) function.  
You can modify it to try any own post-processing methods.  

## Notice points on training and inferece methods  

*Can decode the outputs of SSD network without knowing of its structure?*  
*No!* because to decode outputs you need **information of anchors and two varience values for loss function** at training.  

### anchors data format which must be same at training and inference  

|             | CenterY | CenterX | Height | Width |memo                                           |
|      :-:    |      :-:|      :-:|     :-:|    :-:|:-                                             |
|anchors[i][j]|        0|        1|       2|      3|                                               |
|            0|38, 38, 1|38, 38, 1|4       |4      |coresponding to ssd_300_vgg/block4_box/Reshape |
|            1|19, 19, 1|19, 19, 1|6       |6      |coresponding to ssd_300_vgg/block7_box/Reshape |
|            2|10, 10, 1|10, 10, 1|6       |6      |coresponding to ssd_300_vgg/block8_box/Reshape |
|            3| 5,  5, 1| 5,  5, 1|6       |6      |coresponding to ssd_300_vgg/block9_box/Reshape |
|            4| 3,  3, 1| 3,  3, 1|4       |4      |coresponding to ssd_300_vgg/block10_box/Reshape|
|            5| 1,  1, 1| 1,  1, 1|4       |4      |coresponding to ssd_300_vgg/block11_box/Reshape|

Here, **38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732** default boxes.  

### variance values for loss function which must be same at training and inference  
*varience of location loss: 0.1*  
*varience of size loss: 0.2*  
These are two magic numbers but maybe derived via knowledge of experience for SSD Training.  

## Datasets

The current version only supports Pascal VOC datasets (2007 and 2012). In order to be used for training a SSD model, the former need to be converted to TF-Records using the `tf_convert_data.py` script:
```bash
 $ cd SSD-Tensorflow2
 // After download VOCtest_06-Nov-2007.tar and untar
 $ mkdir ./tfrecords
 $ export DATASET_DIR=./VOCdevkit/VOC2007/
 $ python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_test \
    --output_dir=./tfrecords
```
Note the previous command generated a collection of TF-Records instead of a single file in order to ease shuffling during training.

## Evaluation on Pascal VOC 2007 (Below needs fast CPU or CUDA-GPU)  
- *GPU Environment*  
  OS CentOS7  
  tensorflow-gpu == 1.13.1  
  CUDA cuda_10.0.130_410.48_linux.run  
  cuDNN cudnn-10.0-linux-x64-v7.6.5.32.tgz  

The present TensorFlow implementation of SSD models have the following performances:

| Model | Training data  | Testing data | mAP | FPS  |
|--------|:---------:|:------:|:------:|:------:|
| [SSD-300 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWZlJaRTRRQWRFYXM) | VOC07+12 trainval | VOC07 test | 0.778 | - |
| [SSD-300 VGG-based](https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing) | VOC07+12+COCO trainval | VOC07 test | 0.817 | - |
| [SSD-512 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWT1RCLVZNN3RTVEU) | VOC07+12+COCO trainval | VOC07 test | 0.837 | - |

We are working hard at reproducing the same performance as the original [Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd)!

After downloading and extracting the previous checkpoints, the evaluation metrics should be reproducible by running the following command:
```bash
 // After download model weights and unzip
 $ mkdir ./logs
 $ export EVAL_DIR=./logs/
 $ export CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
 // By python3
 $ python3 eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=./tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1
```
The evaluation script provides estimates on the recall-precision curve and compute the mAP metrics following the Pascal VOC 2007 and 2012 guidelines.

```
...
AP_VOC07/mAP[0.74313211395020939]  // eval with 2007 guildline
AP_VOC12/mAP[0.76659678523265873]  // eval with 2012 guildline
I1105 01:14:30.333214 140431879247616 evaluation.py:275] Finished evaluation at 2019-11-05-01:14:30
```
Notice: *We fitted with tensorflow version 1.13.1(or 1.15.0) following [issue#321](https://github.com/balancap/SSD-Tensorflow/issues/321). In the issue#321 shows how to modify eval_ssd_network.py and tf_extend/metrics.py with 1.13rc1. We made eval_ssd_network.py and tf_extend/metrics.py to fit with our tensorflow versions.*  

In addition, if one wants to experiment/test a different Caffe SSD checkpoint, the former can be converted to TensorFlow checkpoints as following:
```sh
CAFFE_MODEL=./ckpts/SSD_300x300_ft_VOC0712/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel
python caffe_to_tensorflow.py \
    --model_name=ssd_300_vgg \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}
```

## Training

The script `train_ssd_network.py` is in charged of training the network. Similarly to TF-Slim models, one can pass numerous options to the training process (dataset, optimiser, hyper-parameters, model, ...). In particular, it is possible to provide a checkpoint file which can be use as starting point in order to fine-tune a network.

### Fine-tuning existing SSD checkpoints

The easiest way to fine the SSD model is to use as pre-trained SSD network (VGG-300 or VGG-512). For instance, one can fine a model starting from the former as following:
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32
```
Note that in addition to the training script flags, one may also want to experiment with data augmentation parameters (random cropping, resolution, ...) in `ssd_vgg_preprocessing.py` or/and network parameters (feature layers, anchors boxes, ...) in `ssd_vgg_300/512.py`

Furthermore, the training script can be combined with the evaluation routine in order to monitor the performance of saved checkpoints on a validation dataset. For that purpose, one can pass to training and validation scripts a GPU memory upper limit such that both can run in parallel on the same device. If some GPU memory is available for the evaluation script, the former can be run in parallel as follows:
```bash
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

### Fine-tuning a network trained on ImageNet

One can also try to build a new SSD model based on standard architecture (VGG, ResNet, Inception, ...) and set up on top of it the `multibox` layers (with specific anchors, ratios, ...). For that purpose, you can fine-tune a network by only loading the weights of the original architecture, and initialize randomly the rest of network. For instance, in the case of the [VGG-16 architecture](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz), one can train a new model as following:
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./log/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```
Hence, in the former command, the training script randomly initializes the weights belonging to the `checkpoint_exclude_scopes` and load from the checkpoint file `vgg_16.ckpt` the remaining part of the network. Note that we also specify with the `trainable_scopes` parameter to first only train the new SSD components and left the rest of VGG network unchanged. Once the network has converged to a good first result (~0.5 mAP for instance), you can fine-tuned the complete network as following:
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./log_finetune/
CHECKPOINT_PATH=./log/model.ckpt-N
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```

A number of pre-trained weights of popular deep architectures can be found on [TF-Slim models page](https://github.com/tensorflow/models/tree/master/research/slim).
