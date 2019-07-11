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
for i in range(100):
    ip.set_tensor(indexi, img)
    ip.invoke()
    ip.get_tensor(indexo)
print("%.3fFPS"%(100./(time()-start)))
