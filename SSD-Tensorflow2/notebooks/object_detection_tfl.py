#!/usr/bin/env python3
import os, sys
import numpy as np
import cv2
from tflite_runtime import interpreter as tf
from time import time
from random import seed, random
from pdb import set_trace

seed(2223*2)
#VOC label 20+1
class_names = ('__background__',
                 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')
#generate color for class at random
class_color = [(int(random()*255),int(random()*255),int(random()*255)) for i in range(len(class_names)-1)]

# decoder btn prediction and default box
def decode2(pred,priors,var_loc=0.1,var_siz=0.2):
      gx,gy = var_loc*pred[:2]*priors[2:] + priors[:2]
      gw,gh = np.exp(var_siz*pred[2:])*priors[2:]
      return gx,gy,gw,gh

org = cv2.imread('dog.jpg')
org_h, org_w = org.shape[:2]
img = cv2.resize(org, (300,300))
img = img.astype(np.float32)
print("input image size:",org.shape)

#with open('labelmap.txt') as f: LABELS = f.readlines()
#LABELS = [ l.strip() for l in LABELS ]
#background = 1
#LABELS = LABELS[ background: ]
#print("Classes:",len(LABELS))

ip = tf.Interpreter(model_path="./detect.tflite")
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi = [ infoi[i]['index'] for i in range(len(infoi)) ]
indexo = [ infoo[i]['index'] for i in range(len(infoo)) ]
namei  = [ infoi[i]['name' ] for i in range(len(infoi)) ]
nameo  = [ infoo[i]['name' ] for i in range(len(infoo)) ]
assert len(indexi) == 1, 'Unsuppot multiple input tensors'
print("inputs  in pb:{}".format(namei))
print("outputs in pb:{}".format(nameo))

# Spec of outputs of SSD Network
n_classes = len(class_names)
n_boxdims = 4
pred_conf_names = [
"ssd_300_vgg/softmax/Reshape_1",
"ssd_300_vgg/softmax_1/Reshape_1",
"ssd_300_vgg/softmax_2/Reshape_1",
"ssd_300_vgg/softmax_3/Reshape_1",
"ssd_300_vgg/softmax_4/Reshape_1",
"ssd_300_vgg/softmax_5/Reshape_1",
]
pred_hatg_names = [
"ssd_300_vgg/block4_box/Reshape",
"ssd_300_vgg/block7_box/Reshape",
"ssd_300_vgg/block8_box/Reshape",
"ssd_300_vgg/block9_box/Reshape",
"ssd_300_vgg/block10_box/Reshape",
"ssd_300_vgg/block11_box/Reshape",
]
pred_conf_name2tensoridx = { n:indexo[nameo.index(n)] for n in pred_conf_names }
pred_hatg_name2tensoridx = { n:indexo[nameo.index(n)] for n in pred_hatg_names }

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

# invoke inference via SSD Network
ip.set_tensor(indexi[0], img)
ip.invoke()

# get and concatenate result of invoking()
pred_conf = np.concatenate( [ ip.get_tensor(pred_conf_name2tensoridx[i]).reshape(-1,n_classes) for i in pred_conf_names ] )
pred_hatg = np.concatenate( [ ip.get_tensor(pred_hatg_name2tensoridx[i]).reshape(-1,n_boxdims) for i in pred_hatg_names ] )

# setup threshold
flag_background = 1    # which infer image background or not
conf_threshold  = 0.70

# Select class proposals via prediction confidence
ij = np.where(pred_conf[:,flag_background:]>conf_threshold)
prop_classid = np.argmax(pred_conf[ij[0]],axis=1) # prop_classid.shape (8732,21)
prop_names   = {class_names[i] for i in prop_classid}

# Select hat G proposals via prediction confidence
prop_hatg = pred_hatg[ij[0]]                      # prop_hatgs.shape (8732,4)

# Select default box proposals from np_anchors
prop_anchors = np_anchors[ij[0]]

# all outputs are float32 numpy arrays, so convert types as appropriate
boxes   = [ decode2(i, j) for i,j in zip(prop_hatg, prop_anchors) ]
classes = prop_classid
scores  = pred_conf[ij[0]]
Ndets   = prop_hatg.shape[0]
print("num dets: {}".format(Ndets))
print("classes : {}".format(classes))
for i in range(Ndets):
    score       = np.max(scores[i])
    #if score < conf_threshold: continue
    (gx, gy, gw, gh) = boxes[i]
    (left, top, right, bottom) = int(org_w * (gx-gw/2.)), int(org_h * (gy-gh/2.)), int(org_w * (gx+gw/2.)), int(org_h * (gy+gh/2.))
    class_id    = classes[i]
    #set_trace()
    label_txt = "%d-%s"%(class_id,class_names[class_id])
    print("%.3f(%.3d %.3d %.3d %.3d) %d %s"%(score,top,left,bottom,right,class_id,label_txt))
    cv2.rectangle(org,(left,top),(right,bottom),class_color[class_id],1)
    cv2.putText(org,label_txt,(left,top),cv2.FONT_HERSHEY_SIMPLEX,0.5,class_color[class_id],1)
assert cv2.imwrite("result.jpg",org)
cv2.imshow('SSD',org)
while True:
    k = cv2.waitKey(10)
    if k==27:sys.exit(-1)

# For Debug
def view(idx):
    mx=ip.get_tensor(idx).max()
    mn=ip.get_tensor(idx).min()
    me=ip.get_tensor(idx).mean()
    sd=ip.get_tensor(idx).std()
    print("min/max/mean = {:.3f}/{:3f}/{:3f}:{:.6f}".format(mn,mx,me,sd))
