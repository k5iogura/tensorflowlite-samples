# To json from tflite with flatbuffers

How to convert tflite to json text.  

Install flatc command from [google flatbuffers github](https://github.com/google/flatbuffers).  
```
 $ git clone https://github.com/google/flatbuffers
 $ cd flatbuffers; mkdir build; cd build
 $ cmake ..
 $ make
 # make install

 $ which flatc
   /usr/local/bin/flatc
 $ flatc --version
   flatc version 1.11.0
```

Convert tflite to json using tflite schema spec file(schema_v3.fbs).  
```
 $ wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema_v3.fbs
 $ flatc -t schema_v3.fbs -- ./detect.tflite 
 $ ls *.json
   detect.json
```

**July.12,2019**  

