# Conversion from model zoo to tflite  


## Download Model Zoo  

Example is ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03 which includes **tflite_graph.pb** file maybe for *tflite inference*.  
Check [model zoo site](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  
```
 $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
   http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
   Resolving obprx01.intra.hitachi.co.jp (obprx01.intra.hitachi.co.jp)... 158.213.204.12
   Connecting to obprx01.intra.hitachi.co.jp (obprx01.intra.hitachi.co.jp)|158.213.204.12|:8080... connected.
   Proxy request sent, awaiting response... 200 OK
   Length: 144806142 (138M) [application/x-tar]
   Saving to: ‘ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz’
   ssd_mobilenet_v2_quantized_300x300_coco_20 100%[=======================================>] 138.10M  9.35MB/s    in 16s     
   - ‘ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz’ saved [144806142/144806142]

 $ tar xzf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
 $ ls ssd*
   ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

   ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03:
   model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta  pipeline.config  tflite_graph.pb  tflite_graph.pbtxt
```

## Check contents of pb file via node_pb.py  

```
 $ ./node_pb.py ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb
   PATH_TO_FROZEN_GRAPH:ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb
   Traceback (most recent call last):
     File "./node_pb.py", line 25, in <module>
       tf.import_graph_def(od_graph_def, name='')
     File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
       return func(*args, **kwargs)
     File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/importer.py", line 426, in import_graph_def
       graph._c_graph, serialized, options)  # pylint: disable=protected-access
   tensorflow.python.framework.errors_impl.NotFoundError:
   Op type not registered 'TFLite_Detection_PostProcess' in binary running on ub1604lts-shima.
   Make sure the Op and Kernel are registered in the binary running in this process.
   Note that if you are loading a saved graph which used ops from tf.contrib,
   accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily
   registered when the module is first accessed.
```
Failed but don't mind, go to next step.  

## Convert pb to tflite  

Convert pb to tflite with --inference_type=QUANTIZED_UINT8.  
To get correct tensor value via interpreter get_tensor() avoid to reuse local tensor memory area by specifying tensor name with --output_array option such as Squeeze, convert_socres that is input of TFLite_Detection_PostProcess tensor No. are 250, 259.  
```
 $ tflite_convert \
--graph_def_file=ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb \
--output_file=./foo.tflite \
--output_format=TFLITE \
--input_arrays=normalized_input_image_tensor \
--input_shapes=1,300,300,3 \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_dev_values=128 \
--output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3,Squeeze,convert_scores" \
--allow_custom_ops

 $ ls -lh foo.tflite
   -rw-rw-r-- 1 hst20076433 hst20076433 5.9M 10月 25 12:29 foo.tflite
```

## Prepare tflite python directory and check contents of tflite file via node_tfl.py 

You can see contents of tflite file by node_tfl.py.  
```
 $ flatc --python schema_v3+MUL+MAXIMUM.fbs
 $ ls -d tflite/
   tflite
 $ ./node_tfl.py foo.tflite
   ...
   Allocatng Graph ..
   dest_tensor [61] UINT8 FeatureExtractor/MobilenetV2/Conv/Relu6 <= operator CONVD   0(code  2) = src [260, 62, 60]
   dest_tensor [66] UINT8 FeatureExtractor/MobilenetV2/expanded_conv/depthwise/Relu6 <= operator DEPTH   1(code  3) = src [61, 68, 67]
   dest_tensor [70] UINT8 FeatureExtractor/MobilenetV2/expanded_conv/project/add_fold <= operator CONVD   2(code  2) = src [66, 71, 69]
   dest_tensor [76] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_1/expand/Relu6 <= operator CONVD   3(code  2) = src [70, 77, 75]
   dest_tensor [72] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_1/depthwise/Relu6 <= operator DEPTH   4(code  3) = src [76, 74, 73]
   dest_tensor [79] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_1/project/add_fold <= operator CONVD   5(code  2) = src [72, 80, 78]
   dest_tensor [153] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_2/expand/Relu6 <= operator CONVD   6(code  2) = src [79, 154, 152]
   dest_tensor [149] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_2/depthwise/Relu6 <= operator DEPTH   7(code  3) = src [153, 151, 150]
   dest_tensor [156] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_2/project/add_fold <= operator CONVD   8(code  2) = src [149, 157, 155]
   dest_tensor [148] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_2/add <= operator UNKNW   9(code  0) = src [156, 79]
   dest_tensor [162] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_3/expand/Relu6 <= operator CONVD  10(code  2) = src [148, 163, 161]
   dest_tensor [158] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_3/depthwise/Relu6 <= operator DEPTH  11(code  3) = src [162, 160, 159]
   dest_tensor [165] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_3/project/add_fold <= operator CONVD  12(code  2) = src [158, 166, 164]
   dest_tensor [172] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_4/expand/Relu6 <= operator CONVD  13(code  2) = src [165, 173, 171]
   dest_tensor [168] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_4/depthwise/Relu6 <= operator DEPTH  14(code  3) = src [172, 170, 169]
   dest_tensor [175] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_4/project/add_fold <= operator CONVD  15(code  2) = src [168, 176, 174]
   dest_tensor [167] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_4/add <= operator UNKNW  16(code  0) = src [175, 165]
   dest_tensor [182] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_5/expand/Relu6 <= operator CONVD  17(code  2) = src [167, 183, 181]
   dest_tensor [178] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_5/depthwise/Relu6 <= operator DEPTH  18(code  3) = src [182, 180, 179]
   dest_tensor [185] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_5/project/add_fold <= operator CONVD  19(code  2) = src [178, 186, 184]
   dest_tensor [177] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_5/add <= operator UNKNW  20(code  0) = src [185, 167]
   dest_tensor [191] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_6/expand/Relu6 <= operator CONVD  21(code  2) = src [177, 192, 190]
   dest_tensor [187] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_6/depthwise/Relu6 <= operator DEPTH  22(code  3) = src [191, 189, 188]
   dest_tensor [194] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_6/project/add_fold <= operator CONVD  23(code  2) = src [187, 195, 193]
   dest_tensor [201] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_7/expand/Relu6 <= operator CONVD  24(code  2) = src [194, 202, 200]
   dest_tensor [197] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_7/depthwise/Relu6 <= operator DEPTH  25(code  3) = src [201, 199, 198]
   dest_tensor [204] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_7/project/add_fold <= operator CONVD  26(code  2) = src [197, 205, 203]
   dest_tensor [196] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_7/add <= operator UNKNW  27(code  0) = src [204, 194]
   dest_tensor [211] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_8/expand/Relu6 <= operator CONVD  28(code  2) = src [196, 212, 210]
   dest_tensor [207] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_8/depthwise/Relu6 <= operator DEPTH  29(code  3) = src [211, 209, 208]
   dest_tensor [214] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_8/project/add_fold <= operator CONVD  30(code  2) = src [207, 215, 213]
   dest_tensor [206] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_8/add <= operator UNKNW  31(code  0) = src [214, 196]
   dest_tensor [221] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_9/expand/Relu6 <= operator CONVD  32(code  2) = src [206, 222, 220]
   dest_tensor [217] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_9/depthwise/Relu6 <= operator DEPTH  33(code  3) = src [221, 219, 218]
   dest_tensor [224] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_9/project/add_fold <= operator CONVD  34(code  2) = src [217, 225, 223]
   dest_tensor [216] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_9/add <= operator UNKNW  35(code  0) = src [224, 206]
   dest_tensor [85] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_10/expand/Relu6 <= operator CONVD  36(code  2) = src [216, 86, 84]
   dest_tensor [81] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_10/depthwise/Relu6 <= operator DEPTH  37(code  3) = src [85, 83, 82]
   dest_tensor [88] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_10/project/add_fold <= operator CONVD  38(code  2) = src [81, 89, 87]
   dest_tensor [95] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_11/expand/Relu6 <= operator CONVD  39(code  2) = src [88, 96, 94]
   dest_tensor [91] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_11/depthwise/Relu6 <= operator DEPTH  40(code  3) = src [95, 93, 92]
   dest_tensor [98] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_11/project/add_fold <= operator CONVD  41(code  2) = src [91, 99, 97]
   dest_tensor [90] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_11/add <= operator UNKNW  42(code  0) = src [98, 88]
   dest_tensor [105] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_12/expand/Relu6 <= operator CONVD  43(code  2) = src [90, 106, 104]
   dest_tensor [101] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_12/depthwise/Relu6 <= operator DEPTH  44(code  3) = src [105, 103, 102]
   dest_tensor [108] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_12/project/add_fold <= operator CONVD  45(code  2) = src [101, 109, 107]
   dest_tensor [100] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_12/add <= operator UNKNW  46(code  0) = src [108, 90]
   dest_tensor [114] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_13/expand/Relu6 <= operator CONVD  47(code  2) = src [100, 115, 113]
   dest_tensor [0] UINT8 BoxPredictor_0/BoxEncodingPredictor/BiasAdd <= operator CONVD  70(code  2) = src [114, 2, 1]
   dest_tensor [6] UINT8 BoxPredictor_0/Reshape <= operator RESHP  71(code  5) = src [0, 7]
   dest_tensor [110] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_13/depthwise/Relu6 <= operator DEPTH  48(code  3) = src [114, 112, 111]
   dest_tensor [117] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_13/project/add_fold <= operator CONVD  49(code  2) = src [110, 118, 116]
   dest_tensor [124] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_14/expand/Relu6 <= operator CONVD  50(code  2) = src [117, 125, 123]
   dest_tensor [120] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_14/depthwise/Relu6 <= operator DEPTH  51(code  3) = src [124, 122, 121]
   dest_tensor [127] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_14/project/add_fold <= operator CONVD  52(code  2) = src [120, 128, 126]
   dest_tensor [119] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_14/add <= operator UNKNW  53(code  0) = src [127, 117]
   dest_tensor [134] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_15/expand/Relu6 <= operator CONVD  54(code  2) = src [119, 135, 133]
   dest_tensor [130] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_15/depthwise/Relu6 <= operator DEPTH  55(code  3) = src [134, 132, 131]
   dest_tensor [137] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_15/project/add_fold <= operator CONVD  56(code  2) = src [130, 138, 136]
   dest_tensor [129] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_15/add <= operator UNKNW  57(code  0) = src [137, 119]
   dest_tensor [143] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_16/expand/Relu6 <= operator CONVD  58(code  2) = src [129, 144, 142]
   dest_tensor [139] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_16/depthwise/Relu6 <= operator DEPTH  59(code  3) = src [143, 141, 140]
   dest_tensor [146] UINT8 FeatureExtractor/MobilenetV2/expanded_conv_16/project/add_fold <= operator CONVD  60(code  2) = src [139, 147, 145]
   dest_tensor [64] UINT8 FeatureExtractor/MobilenetV2/Conv_1/Relu6 <= operator CONVD  61(code  2) = src [146, 65, 63]
   dest_tensor [10] UINT8 BoxPredictor_1/BoxEncodingPredictor/BiasAdd <= operator CONVD  74(code  2) = src [64, 12, 11]
   dest_tensor [16] UINT8 BoxPredictor_1/Reshape <= operator RESHP  75(code  5) = src [10, 17]
   dest_tensor [227] UINT8 FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_2_1x1_256/Relu6 <= operator CONVD  62(code  2) = src [64, 228, 226]
   dest_tensor [239] UINT8 FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_512/Relu6 <= operator CONVD  63(code  2) = src [227, 240, 238]
   dest_tensor [20] UINT8 BoxPredictor_2/BoxEncodingPredictor/BiasAdd <= operator CONVD  78(code  2) = src [239, 22, 21]
   dest_tensor [26] UINT8 BoxPredictor_2/Reshape <= operator RESHP  79(code  5) = src [20, 27]
   dest_tensor [230] UINT8 FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_3_1x1_128/Relu6 <= operator CONVD  64(code  2) = src [239, 231, 229]
   dest_tensor [242] UINT8 FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_3_3x3_s2_256/Relu6 <= operator CONVD  65(code  2) = src [230, 243, 241]
   dest_tensor [30] UINT8 BoxPredictor_3/BoxEncodingPredictor/BiasAdd <= operator CONVD  82(code  2) = src [242, 32, 31]
   dest_tensor [36] UINT8 BoxPredictor_3/Reshape <= operator RESHP  83(code  5) = src [30, 37]
   dest_tensor [233] UINT8 FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_4_1x1_128/Relu6 <= operator CONVD  66(code  2) = src [242, 234, 232]
   dest_tensor [245] UINT8 FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_4_3x3_s2_256/Relu6 <= operator CONVD  67(code  2) = src [233, 246, 244]
   dest_tensor [40] UINT8 BoxPredictor_4/BoxEncodingPredictor/BiasAdd <= operator CONVD  86(code  2) = src [245, 42, 41]
   dest_tensor [46] UINT8 BoxPredictor_4/Reshape <= operator RESHP  87(code  5) = src [40, 47]
   dest_tensor [236] UINT8 FeatureExtractor/MobilenetV2/layer_19_1_Conv2d_5_1x1_64/Relu6 <= operator CONVD  68(code  2) = src [245, 237, 235]
   dest_tensor [248] UINT8 FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/Relu6 <= operator CONVD  69(code  2) = src [236, 249, 247]
   dest_tensor [50] UINT8 BoxPredictor_5/BoxEncodingPredictor/BiasAdd <= operator CONVD  90(code  2) = src [248, 52, 51]
   dest_tensor [56] UINT8 BoxPredictor_5/Reshape <= operator RESHP  91(code  5) = src [50, 57]
   dest_tensor [257] UINT8 concat           <= operator CONCT  92(code  1) = src [6, 16, 26, 36, 46, 56]
   dest_tensor [250] UINT8 Squeeze          <= operator RESHP  93(code  5) = src [257, 251]
   dest_tensor [3] UINT8 BoxPredictor_0/ClassPredictor/BiasAdd <= operator CONVD  72(code  2) = src [114, 5, 4]
   dest_tensor [8] UINT8 BoxPredictor_0/Reshape_1 <= operator RESHP  73(code  5) = src [3, 9]
   dest_tensor [13] UINT8 BoxPredictor_1/ClassPredictor/BiasAdd <= operator CONVD  76(code  2) = src [64, 15, 14]
   dest_tensor [18] UINT8 BoxPredictor_1/Reshape_1 <= operator RESHP  77(code  5) = src [13, 19]
   dest_tensor [23] UINT8 BoxPredictor_2/ClassPredictor/BiasAdd <= operator CONVD  80(code  2) = src [239, 25, 24]
   dest_tensor [28] UINT8 BoxPredictor_2/Reshape_1 <= operator RESHP  81(code  5) = src [23, 29]
   dest_tensor [33] UINT8 BoxPredictor_3/ClassPredictor/BiasAdd <= operator CONVD  84(code  2) = src [242, 35, 34]
   dest_tensor [38] UINT8 BoxPredictor_3/Reshape_1 <= operator RESHP  85(code  5) = src [33, 39]
   dest_tensor [43] UINT8 BoxPredictor_4/ClassPredictor/BiasAdd <= operator CONVD  88(code  2) = src [245, 45, 44]
   dest_tensor [48] UINT8 BoxPredictor_4/Reshape_1 <= operator RESHP  89(code  5) = src [43, 49]
   dest_tensor [53] UINT8 BoxPredictor_5/ClassPredictor/BiasAdd <= operator CONVD  94(code  2) = src [248, 55, 54]
   dest_tensor [58] UINT8 BoxPredictor_5/Reshape_1 <= operator RESHP  95(code  5) = src [53, 59]
   dest_tensor [258] UINT8 concat_1         <= operator CONCT  96(code  1) = src [8, 18, 28, 38, 48, 58]
   dest_tensor [259] UINT8 convert_scores   <= operator LOGST  97(code  4) = src [258]
   dest_tensor [252, 253, 254, 255] FLOAT32 TFLite_Detection_PostProcess <= operator CUSTM  98(code  6) = src [250, 259, 256]
   Allocatng Graph done.
```

