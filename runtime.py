#!/usr/bin/env python3
import os, sys
import numpy as np
import cv2
from tflite_runtime import interpreter as tf
from time import time

org = cv2.imread('dog.jpg')
org_h, org_w = org.shape[:2]
img = cv2.resize(org, (300,300))
img = img[np.newaxis,:,:,:]
print("input image size:",org.shape)

with open('labelmap.txt') as f: LABELS = f.readlines()
LABELS = [ l.strip() for l in LABELS ]
LABELS = LABELS[1:]
#for j,l in enumerate(LABELS):print(j,":",l)
print("Classes:",len(LABELS))

ip = tf.Interpreter(model_path="./detect.tflite")
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
# get initial value
v85 = ip.get_tensor(85)
i86 = ip.get_tensor(86)
i84 = ip.get_tensor(84)
ip.set_tensor(86, np.full((24,3,3,3),152,dtype=np.uint8))   # Filter
ip.set_tensor(84, np.full((24)      ,0,np.int32))     # Bias
ip.set_tensor(175,np.full(img.shape,0,np.uint8))
ip.invoke()

# reset and infer
i85 = ip.get_tensor(85)
np.save('i85.npy',i85)
ip.set_tensor(85, i85)
ip.set_tensor(86, i86)
ip.set_tensor(84, i84)
ip.set_tensor(175, img)
ip.invoke()

boxes   = ip.get_tensor(indexo+0)
classes = ip.get_tensor(indexo+1)
scores  = ip.get_tensor(indexo+2)
Ndets   = ip.get_tensor(indexo+3)
print("location:",boxes)
print("classes :",classes)
print("score   :",scores)
print("num dets:",Ndets)
print("%.3fFPS"%(1./(time()-start)))
for i in range(int(Ndets[0])):
    score       = scores[0][i]
    if score < 0.6: continue
    (top, left, bottom, right) = boxes[0][i]
    tl = (int(left*org_w),  int(top*org_h))
    rb = (int(right*org_w), int(bottom*org_h))
    class_id    = int(classes[0][i])
    print("%.3f(%.3f %.3f %.3f %.3f) %d"%(score,tl[0],tl[1],rb[0],rb[1],class_id))
    label_txt = "%d-%s"%(class_id,LABELS[class_id])
    cv2.rectangle(org,tl,rb,(255,255,255),1)
    cv2.putText(org,label_txt,tl,cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
cv2.imwrite("result.jpg",org)

print("t85 <= 175 84 86")
t85=ip.get_tensor(85)
t175=ip.get_tensor(175)
t84=ip.get_tensor(84)
t86=ip.get_tensor(86)

print(set(np.where(t85>0)[3]))
