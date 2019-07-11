import numpy as np
import cv2
import tensorflow as tf
from time import time

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (300,300))
img = img[np.newaxis,:,:,:]

ip = tf.lite.Interpreter(model_path="./detect.tflite")
ip.allocate_tensors()

infoi=ip.get_input_details()
infoo=ip.get_output_details()

indexi=infoi[0]['index']
indexo=infoo[0]['index']

start=time()
ip.set_tensor(indexi, img)
ip.invoke()
boxes   = ip.get_tensor(indexo+0)
classes = ip.get_tensor(indexo+1)
scores  = ip.get_tensor(indexo+2)
Ndets   = ip.get_tensor(indexo+3)
print("location:",boxes)
print("classes :",classes)
print("score   :",scores)
print("num dets:",Ndets)
print("%.3fFPS"%(10./(time()-start)))
for i in range(int(Ndets[0])):
    score       = scores[0][i]
    if score < 0.6: continue
    (top, left, bottom, right) = boxes[0][i]
    rect = ( left, top, right, bottom)
    class_id    = int(classes[0][i])
    print("%.3f(%.3f %.3f %.3f %.3f) %d"%(score,rect[0],rect[1],rect[2],rect[3],class_id))

