# -*- coding: utf-8 -*-
from pdb import set_trace
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from glob import glob
from random import random, seed

seed(2223*2)
#VOC label 20+1
class_names = ('__background__',
                 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')
#generate color for class at random
class_color = [(int(random()*255),int(random()*255),int(random()*255)) for i in range(len(class_names)-1)]

from pdb import set_trace

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# What model to download.
MODEL_NAME = '.'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/ssd_net_frozen.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

# Make default graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# image select to feed to model
PATH_TO_TEST_IMAGES_DIR = '../demo/*.jpg'
TEST_IMAGE_PATHS = glob(PATH_TO_TEST_IMAGES_DIR)

# Generates ssd_anchors as default box
#
#  center-Y  ,  center-X  ,  H  ,  W   <= Notice!
#[(38, 38, 1), (38, 38, 1), (4,), (4,)]
#[(19, 19, 1), (19, 19, 1), (6,), (6,)]
#[(10, 10, 1), (10, 10, 1), (6,), (6,)]
#[( 5,  5, 1), ( 5,  5, 1), (6,), (6,)]
#[( 3,  3, 1), ( 3,  3, 1), (4,), (4,)]
#[( 1,  1, 1), ( 1,  1, 1), (4,), (4,)]
from nets import ssd_vgg_300
ssd_net = ssd_vgg_300.SSDNet()
net_shape = (300, 300)
ssd_anchors = ssd_net.anchors(net_shape)   # anchors function includes only numpy
np_anchors = []
for i in range(len(ssd_anchors)):
      for cx, cy in   zip(ssd_anchors[i][0].reshape(-1), ssd_anchors[i][1].reshape(-1)):
          for w, h in zip(ssd_anchors[i][2].reshape(-1), ssd_anchors[i][3].reshape(-1)):
              np_anchors.append([cy,cx,h,w]) # the order in anchors are (center-Y, center-X, Height, Width)
np_anchors = np.asarray(np_anchors)        # np_anchors.shape (8732,4)

def run_inference_for_single_image(image, anchors, graph):
    with graph.as_default():
      with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
         # "ExpandDims",
          "ssd_300_vgg/block4_box/Reshape",
          "ssd_300_vgg/block7_box/Reshape",
          "ssd_300_vgg/block8_box/Reshape",
          "ssd_300_vgg/block9_box/Reshape",
          "ssd_300_vgg/block10_box/Reshape",
          "ssd_300_vgg/block11_box/Reshape",
          "ssd_300_vgg/softmax/Reshape_1",
          "ssd_300_vgg/softmax_1/Reshape_1",
          "ssd_300_vgg/softmax_2/Reshape_1",
          "ssd_300_vgg/softmax_3/Reshape_1",
          "ssd_300_vgg/softmax_4/Reshape_1",
          "ssd_300_vgg/softmax_5/Reshape_1"
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image})
        # Select class proposals via prediction confidence
        n_classes = len(class_names)
        n_boxdims = 4
        pred_conf = np.concatenate(
            [np.reshape(output_dict["ssd_300_vgg/softmax/Reshape_1"  ][0],(-1,n_classes)),
            np.reshape(output_dict[ "ssd_300_vgg/softmax_1/Reshape_1"][0],(-1,n_classes)),
            np.reshape(output_dict[ "ssd_300_vgg/softmax_2/Reshape_1"][0],(-1,n_classes)),
            np.reshape(output_dict[ "ssd_300_vgg/softmax_3/Reshape_1"][0],(-1,n_classes)),
            np.reshape(output_dict[ "ssd_300_vgg/softmax_4/Reshape_1"][0],(-1,n_classes)),
            np.reshape(output_dict[ "ssd_300_vgg/softmax_5/Reshape_1"][0],(-1,n_classes))],
            axis=0)
        flag_background = 1    # which infer image background or not
        conf_threshold  = 0.70
        ij = np.where(pred_conf[:,flag_background:]>conf_threshold)
        prop_classid = np.argmax(pred_conf[ij[0]],axis=1) # prop_classid.shape (8732,21)
        prop_names   = {class_names[i] for i in prop_classid}

        # Select hat G proposals via prediction confidence
        pred_hatgs = np.concatenate(
            [np.reshape(output_dict["ssd_300_vgg/block4_box/Reshape" ][0],(-1,n_boxdims)),
            np.reshape(output_dict[ "ssd_300_vgg/block7_box/Reshape" ][0],(-1,n_boxdims)),
            np.reshape(output_dict[ "ssd_300_vgg/block8_box/Reshape" ][0],(-1,n_boxdims)),
            np.reshape(output_dict[ "ssd_300_vgg/block9_box/Reshape" ][0],(-1,n_boxdims)),
            np.reshape(output_dict[ "ssd_300_vgg/block10_box/Reshape"][0],(-1,n_boxdims)),
            np.reshape(output_dict[ "ssd_300_vgg/block11_box/Reshape"][0],(-1,n_boxdims))],
            axis=0)
        prop_hatgs = pred_hatgs[ij[0]]                     # prop_hatgs.shape (8732,4)

        # Select default box proposals from np_anchors
        prop_anchors = anchors[ij[0]]

        # all outputs are float32 numpy arrays, so convert types as appropriate
        ret_dict={}
        ret_dict['num_prop']    = prop_hatgs.shape[0]
        ret_dict['prop_hatgs']  = prop_hatgs
        ret_dict['prop_scores'] = pred_conf[ij[0]]
        ret_dict['prop_anchors']= prop_anchors
    return ret_dict


def decode1(i,pred,priors,var_loc=0.1,var_siz=0.2):
      gx,gy = var_loc*pred[0,i,:2]*priors[i,2:] + priors[i,:2]
      gw,gh = np.exp(var_siz*pred[0,i,2:])*priors[i,2:]
      return gx,gy,gw,gh

def decode2(pred,priors,var_loc=0.1,var_siz=0.2):
      gx,gy = var_loc*pred[:2]*priors[2:] + priors[:2]
      gw,gh = np.exp(var_siz*pred[2:])*priors[2:]
      return gx,gy,gw,gh

# Start inference for images
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_bgr= cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, np_anchors, detection_graph)

    org_h, org_w = image_np.shape[:2]

    # Draw result info on a input image
    for i, (hatg, prior) in enumerate(zip(output_dict['prop_hatgs'], output_dict['prop_anchors'])):

        # convert score proposals to class name
        classid = np.argmax(output_dict['prop_scores'][i])
        classnm = class_names[classid]
        score   = output_dict['prop_scores'][i][classid]*100.

        # convert hatG and prior box to predicted box
        gx,gy,gw,gh = decode2(hatg, prior)
        lx,ly,rx,ry = int(org_w * (gx-gw/2.)), int(org_h * (gy-gh/2.)), int(org_w * (gx+gw/2.)), int(org_h * (gy+gh/2.))

        # draw result
        print('{:12s}({:3d}) {:2d}% {:4d} {:4d} {:4d} {:4d}'.format(classnm, classid, int(score), lx, ly, rx, ry))
        image_bgr = cv2.rectangle(image_bgr,(lx,ly),(rx,ry),class_color[classid],1)

    cv2.imshow('demo',image_bgr)
    while True:
      k=cv2.waitKey(10)
      if k==27:sys.exit(-1)
      if k!=-1:break
    print("Next",k)

