## [TensorFlow Lite Interpreter get_tensor() #23384](https://github.com/tensorflow/tensorflow/issues/23384)  

There are no guaranty
**Describe the problem**  
"When trying to write out tensors to file using the TensorFlow Lite Interpreter::get_tensor() function, mostly incorrect data is being returned. For the attached input, gray128.jpg, an image with all pixels set to RGB(128, 128, 128), I expect the layer outputs to be fairly repetitive, but it is not.  

To further narrow down the issue, I modified the bias and weight tensors for the first layer to 0.0. After convolution and activation, I would expect the output tensor to be completely 0's, but it is not. The output is consistent across runs (deterministic). The input, output, bias, and weight tensors all seem to be written out correctly, but most of the intermediate output tensors do not seem to be.  

I am doing this to try and verify the intermediate outputs with my own calculations. I was hoping to get inception verified with floating point, then with the uint8 quantized model, then with my own model."  

**Reason**  
"After seeing [this answer](https://stackoverflow.com/a/53105809) on SO about it, I guess intermediate tensors are not guaranteed to have useful data, only the graph output(s). That makes sense for memory reasons, so I probably shouldn't expect this to work anytime soon."  

**Mostly Answer**
"I guess intermediate tensors are **not guaranteed to have useful data, only the graph output(s).**"  

[I uploaded the hack-y modified label_image.py script I used to inspect intermediate tensors here](https://github.com/raymond-li/tflite_tensor_outputter)  

**Synopsis**  
tflite_tensor_outputter.py modifys output tensor index of network in tflite binary file. After modified and invoke(), you can retrieve correct tensor vlaues.   

### My solution  
Transform binary tflite to json format and change output tensor index to need in json file and re-compile it to binary tflite file instead of directory binary tflite file editing.  
```
 $ flatc --strict-json -t schema_v3.fbs -- some.tflite  // transform to json
 $ ls some.json
   some.json
 $ vi some.json
   // modity outputs tensor index of graph to needed one
 $ mv some.tflite some.tflite.back                     // backup if needed
 $ flatc -b -c schema_v3.fbs some.json                 // re-compile to binary tflite
 $ ls some.tflite
   some.tflite
```
Can see correct tensor values needed.  
