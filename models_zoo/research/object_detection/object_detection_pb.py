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
from object_detection_utils import *
from pdb import set_trace

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

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

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)



def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'Postprocessor/ExpandDims',
          'concat','concat_1',
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

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
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  c0=output_dict['pred']
  c1=output_dict['conf']
  c2=output_dict['anch'][0]
  s1=softmax(c1[...,1:])
  idxs = np.where(s1>0.95)
  set_trace()
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

