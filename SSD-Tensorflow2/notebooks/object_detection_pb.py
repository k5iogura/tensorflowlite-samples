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

#VOC label 20+1
class_names = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

#from object_detection_utils import *
from pdb import set_trace

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops

#if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#from object_detection.utils import label_map_util

# from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = '.'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/ssd_net_frozen.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '../demo/*.jpg'
TEST_IMAGE_PATHS = glob(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Generates anchors
from nets import ssd_vgg_300
ssd_net = ssd_vgg_300.SSDNet()
net_shape = (300, 300)
ssd_anchors = ssd_net.anchors(net_shape) # anchors function includes only numpy

def run_inference_for_single_image(image, graph):
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
      conf_threshold  = 0.95
      ij = np.where(pred_conf[:,flag_background:]>conf_threshold)
      prop_classid = np.argmax(pred_conf[ij[0]],axis=1)
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
      prop_hatgs = pred_hatgs[ij[0]]
      set_trace()

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      output_dict['pred'] = output_dict['concat']
      output_dict['conf'] = output_dict['concat_1']
      output_dict['anch'] = output_dict['Postprocessor/ExpandDims']
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def decode(i,pred,priors,var_loc=0.1,var_siz=0.2):
    gx,gy = var_loc*pred[0,i,:2]*priors[i,2:] + priors[i,:2]
    gw,gh = np.exp(var_siz*pred[0,i,2:])*priors[i,2:]
    return gx,gy,gw,gh

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_np_expanded = image_np
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  c0=output_dict['pred']
  c1=output_dict['conf']
  c2=output_dict['anch'][0]
  s1=softmax(c1[...,1:])
  idxs = np.where(s1>0.95)
  high_conf_regions = c0[0, idxs[1],]
  high_conf_classes = c1[0, idxs[1],]
  high_conf_anchors = np.clip(c2[idxs[1],],0., 1.)
  #                defaultBox_wh             * p_xy                      + defaultBox_xy
  gxy_pred = 0.1 * high_conf_anchors[...,2:] * high_conf_regions[...,:2] + high_conf_anchors[...,:2]
  #                defaultBox_wh             * np.exp( p_wh )
  gwh_pred = high_conf_anchors[...,2:] * np.exp(0.2 * high_conf_regions[...,2:])
  org_h, org_w = image_np.shape[:2]
  for j in idxs[1]:
      #xy = c0[0, j, :2]
      #wh = c0[0, j, 2:]
      gx,gy,gw,gh = np.clip(decode(j, c0, c2),0.,1.)
      classid = np.argmax(s1[0,j])
      cx,cy,w,h = c2[j,]
      lx, ly, rx, ry = int(org_w*(cx-w/2.)), int(org_h*(cy-h/2.)), int(org_w*(cx+w/2.)), int(org_h*(cy+h/2.))
      #lx,ly,rx,ry = int(org_w * (gx-gw/2.)), int(org_h * (gy-gh/2.)), int(org_w * (gx+gw/2.)), int(org_h * (gy+gh/2.))
      #lx=0 if lx<0 else lx
      #ly=0 if ly<0 else ly
      #if lx<0 or ly<0:continue
      print(lx,ly,rx,ry)
      if classid == 17: image_np = cv2.rectangle(image_np,(lx,ly),(rx,ry),(0,0,255),1)
      else: image_np = cv2.rectangle(image_np,(lx,ly),(rx,ry),(255,255,255),1)
      #set_trace()
  cv2.imshow('demo',image_np)
  while True:
    if cv2.waitKey(10)!=-1:break
  set_trace()
  sys.exit(-1)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  #plt.figure(figsize=IMAGE_SIZE)
  #plt.imshow(image_np)
  cv2.imshow('image',cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  print(image_path)
  while True:
    if cv2.waitKey(1)!=-1:break
  print("Next")

