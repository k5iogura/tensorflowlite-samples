#!/usr/bin/env python3
import os, sys
import numpy as np
import cv2
import tensorflow as tf
from time import time

cap = cv2.VideoCapture(0)
assert cap is not None

# setup labels
with open('labelmap.txt') as f: LABELS = f.readlines()
LABELS = [ l.strip() for l in LABELS ]
LABELS = LABELS[1:]
for j,l in enumerate(LABELS):print(j,":",l)
print("Classes:",len(LABELS))

# setup interpreter
ip = tf.lite.Interpreter(model_path="./detect.tflite")
ip.allocate_tensors()

# network infomations
infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
images = 0
while True:
    images+=1
    r,org = cap.read()
    assert r
    res = org
    org_h, org_w = org.shape[:2]
    img = cv2.resize(org, (300,300))
    img = img[np.newaxis,:,:,:]

    ip.set_tensor(indexi, img)
    ip.invoke()
    boxes   = ip.get_tensor(indexo+0)
    classes = ip.get_tensor(indexo+1)
    scores  = ip.get_tensor(indexo+2)
    Ndets   = ip.get_tensor(indexo+3)
    for i in range(int(Ndets[0])):
        score       = scores[0][i]
        if score < 0.6: continue
        (top, left, bottom, right) = boxes[0][i]
        tl = (int(left*org_w),  int(top*org_h))
        rb = (int(right*org_w), int(bottom*org_h))
        class_id    = int(classes[0][i])
        label_txt = "%d-%s"%(class_id,LABELS[class_id])
        res = cv2.rectangle(org,tl,rb,(255,255,255),1)
        res = cv2.putText(res,label_txt,tl,cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
    cv2.imshow('detect',res)
    if cv2.waitKey(1) != -1:break
    sys.stdout.write('\b'*40)
    sys.stdout.write('%.3fFPS'%(images/(time()-start)))
    sys.stdout.flush()

print("\nfin")
cv2.destroyAllWindows()
cap.release()

