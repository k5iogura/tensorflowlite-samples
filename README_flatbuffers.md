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

Install python I/F.  
```
 $ pip install python
 $ python -c "import flatbuffers"
 $
```

Convert tflite to json using tflite schema spec file(schema_v3.fbs).  
```
 $ wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema_v3.fbs
 $ flatc --strict-json -t schema_v3.fbs -- ./detect.tflite 
 $ ls *.json
   detect.json
```

About json of detect.tflite flatbuffers with scheme_v3.fbs.  
```
[
  operator_codes:[
  { builtin_code: "CONV_2D" },   <== operator_code_index 0
  ...
  { builtin_code: "CUSTOM",
    custom_code : "TFLite_Detection_PostProcess" }
  ]  
  subgraphs:[  
    { tensors:[
      { shape:[1,19,19,12],      <== tensors_index 0      
        type:  "UINT8",          
        buffer: 93,              <== buffers_index(points buffers area)  
        name:  "BoxPredictor_0/BoxEncodingPredictor/BiasAdd",
        quantization:[
          min:[-15.094324],max:[5.057784],scale[0.079028],zero_point:[191]
        ] },     
      ...
      { shape:[ ... ], type: ... }
      ],
      inputs: [tensors_index, ... ],    
      outputs:[tensors_index, ... ],    
      operators:[
      {
        inputs: [tensors_index, ... ],
        outputs:[tensors_index, ... ],
        builtin_options_type: "Conv2DOptions",
        builtin_options:{stride_w: 2, stride_h: 2},
      },
      ...
      {
        opcode_index: 5          <== operator_code_index
        inputs: [tensors_index, ... ],
        outputs:[tensors_index, ... ],
        custom_options: [121, ... ,1]
      ] }
    }
  ],
  buffers:[  
    data:[204, ... ,255],    <== buffers_index 0   
    ...
    data:[153, ... ,237]
  ]
]
```

**July.12,2019**  

