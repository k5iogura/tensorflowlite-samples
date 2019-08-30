## [TensorFlow Lite Interpreter get_tensor() #23384](https://github.com/tensorflow/tensorflow/issues/23384)  

**Describe the problem**  

When trying to write out tensors to file using the TensorFlow Lite Interpreter::get_tensor() function, mostly incorrect data is being returned. For the attached input, gray128.jpg, an image with all pixels set to RGB(128, 128, 128), I expect the layer outputs to be fairly repetitive, but it is not.  

To further narrow down the issue, I modified the bias and weight tensors for the first layer to 0.0. After convolution and activation, I would expect the output tensor to be completely 0's, but it is not. The output is consistent across runs (deterministic). The input, output, bias, and weight tensors all seem to be written out correctly, but most of the intermediate output tensors do not seem to be.  

I am doing this to try and verify the intermediate outputs with my own calculations. I was hoping to get inception verified with floating point, then with the uint8 quantized model, then with my own model.  
