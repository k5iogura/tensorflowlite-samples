/*
The MIT License (MIT)

Copyright (c) 2019 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <queue>
#include <iterator>
#include <fstream>
#include <numeric>
#include <random>

#include "mnist-loader.h"

#ifdef __clang__
#pragma clang diagnostic push "-Weverything"
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h" // PrintInterpreterState

#ifdef __clang__
#pragma clang diagnostic pop
#endif

static std::string JoinPath(const std::string &dir, const std::string &filename) {
  if (dir.empty()) {
    return filename;
  } else {
    // check '/'
    char lastChar = *dir.rbegin();
    if (lastChar != '/') {
      return dir + std::string("/") + filename;
    } else {
      return dir + filename;
    }
  }
}

// ubyte to float
static void ConvertImage(const uint8_t *src, const float scale,
                         const float offset,
                         float *dst) {
    // Assume memory has been already allocated for `dst`
    for (size_t i = 0; i < 28 * 28; i++) {
        dst[i] = scale * float(src[i]) + offset;
    }
}

// From tensorflow lite repo ==================================================

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*static void PrintProfilingInfo(const tflite::profiling::ProfileEvent* e, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  std::cout << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Node " << std::setw(3) << std::setprecision(3) << op_index
            << ", OpCode " << std::setw(3) << std::setprecision(3)
            << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<tflite::BuiltinOperator>(registration.builtin_code))
            << "\n";
}*/

// ============================================================================

static bool InspectModel(const std::string &model_filename)
{
  std::ifstream f(model_filename);
  if (!f) {
    std::cerr << "Failed to read file: " << model_filename << std::endl;
    return false;
  }

  f.seekg(0, f.end);
  size_t sz = static_cast<size_t>(f.tellg());
  f.seekg(0, f.beg);

  std::vector<uint8_t> buf;
  buf.resize(sz);
  f.read(reinterpret_cast<char *>(buf.data()),
         static_cast<std::streamsize>(sz));

  const tflite::Model *_model = tflite::GetModel(reinterpret_cast<void *>(buf.data()));

  std::cout << "# of buffers = " << _model->buffers()->size() << "\n";
  for (uint32_t i = 0; i < uint32_t(_model->buffers()->size()); i++) {
    const tflite::Buffer *_buffer = _model->buffers()->Get(flatbuffers::uoffset_t(i));
    if (_buffer->data()) {
      std::cout << "[" << i << "] size = " << _buffer->data()->size() << "\n";
    } else {
      std::cout << "[" << i << "] empty" << "\n";
    }
  }

  const tflite::SubGraph *_subgraph = _model->subgraphs()->Get(0);

  std::cout << "# of tensors = " << _subgraph->tensors()->size() << "\n";
  for (uint32_t i = 0; i < uint32_t(_subgraph->tensors()->size()); i++) {
    const tflite::Tensor *_tensor = _subgraph->tensors()->Get(flatbuffers::uoffset_t(i));
    std::cout << "[" << i << "] is_variable = " << _tensor->is_variable() << "\n";
    std::cout << "[" << i << "] buffer = " << _tensor->buffer() << "\n";
    for (uint32_t s = 0; s < uint32_t(_tensor->shape()->size()); s++) {
      std::cout << "[" << i << "] shape[" << s << "] = " << _tensor->shape()->Get(flatbuffers::uoffset_t(s)) << "\n";
    }
  }


  return true;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "mnist (mnist.tflite) (path_to_minist_data) (num_threads)" << std::endl;
  }

  std::string input_modelfilename = "../mnist.tflite";
  if (argc > 1) {
    input_modelfilename = argv[1];
  }

  std::string mnist_data_path = "../data";
  if (argc > 2) {
    mnist_data_path = argv[2];
  }

  int num_threads = -1;
  if (argc > 3) {
    num_threads = std::stoi(argv[3]);
  }

  std::cout << "Input model filename : " << input_modelfilename << "\n";
  std::cout << "MNIST data path      : " << mnist_data_path << "\n";

  // DBG
  if (!InspectModel(input_modelfilename)) {
    return EXIT_FAILURE;
  }

  // Load MNIST data
  std::string mnist_label_filename = JoinPath(mnist_data_path, "t10k-labels-idx1-ubyte");
  std::string mnist_image_filename = JoinPath(mnist_data_path, "t10k-images-idx3-ubyte");
  std::vector<uint8_t> mnist_images;
  std::vector<int> mnist_labels;
  if (!example::LoadMNISTData(mnist_label_filename,
                   mnist_image_filename,
                   &mnist_images, &mnist_labels)) {
    std::cerr << "Failed to load MNIST data.\n" << std::endl;
    return EXIT_FAILURE;
  }



  std::unique_ptr<tflite::FlatBufferModel> model;
  //model = tflite::FlatBufferModel::VerifyAndBuildFromFile(input_modelfilename.c_str());
  model = tflite::FlatBufferModel::BuildFromFile(input_modelfilename.c_str());
  if (!model) {
    std::cerr << "failed to load model. filename = " << input_modelfilename << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Loaded model : " << input_modelfilename << "\n";
  model->error_reporter();

  tflite::ops::builtin::BuiltinOpResolver resolver;

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    std::cerr << "failed to construct interpreter.\n";
    return EXIT_FAILURE;
  }

  // optional
  // interpreter->UseNNAPI(1);

  {
    std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
    std::cout << "inputs: " << interpreter->inputs().size() << "\n";
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = int(interpreter->tensors_size());
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (num_threads > 0) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];
  std::cout << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  std::cout << "number of inputs: " << inputs.size() << "\n";
  std::cout << "number of outputs: " << outputs.size() << "\n";

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors!\n";
    return EXIT_FAILURE;
  }

  std::cout << "============================\n";
  tflite::PrintInterpreterState(interpreter.get());
  std::cout << "============================\n";

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  if (dims->size != 2) {
    std::cerr << "dim must be 2.\n";
    return EXIT_FAILURE;
  }

  int wanted_size = dims->data[1];
  if (wanted_size != (28 * 28)) {
    std::cerr << "Invalid input tensor shape\n";
    return EXIT_FAILURE;
  }

  if (interpreter->tensor(input)->type != kTfLiteFloat32) {
    std::cerr << "Supportly only Float32 type for input\n";
    return EXIT_FAILURE;
  }

  int output = interpreter->outputs()[0];
  if (interpreter->tensor(output)->type != kTfLiteFloat32) {
    std::cerr << "Supportly only Float32 type for output\n";
    return EXIT_FAILURE;
  }


  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  if (output_size != 10) {
    std::cerr << "output size is not 10\n";
    return EXIT_FAILURE;
  }

  // Run inference
  //tflite::profiling::Profiler profiler;
  {
    //interpreter->SetProfiler(&profiler);

    double sum = 0.0;

    // generate [0, 10000) sequence.
    std::vector<int> indices(10000);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::shuffle(indices.begin(), indices.end(), engine);

    // Shrink to get randomly choosen `count` indices.
    size_t count = 7777;
    assert(count <= 10000);
    indices.resize(count);

    for (size_t _i = 0; _i < count; _i++) {

      size_t i = size_t(indices[_i]);

      //profiler.StartProfiling();

      ConvertImage(mnist_images.data() +  i * 28 * 28, 1.0f / 255.0f, 0.0f,
        interpreter->typed_tensor<float>(input));

      if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite!\n";
        return EXIT_FAILURE;
      }

      //profiler.StopProfiling();

      // argmax
      float *best = std::max_element(interpreter->typed_output_tensor<float>(0), interpreter->typed_output_tensor<float>(0) + 10);
      int digit = int(std::distance(interpreter->typed_output_tensor<float>(0), best));
      assert(digit >= 0);
      assert(digit < 10);

      std::cout << (digit == mnist_labels[i]) << ", " << i << ", estimated = " << digit << ", answer = " << mnist_labels[i] << "\n";

      sum += (digit == mnist_labels[i]) ? 1.0 : 0.0;
    }

    std::cout << "Accuray = " << 100.0 * sum / double(count) << " %%\n";

    /*auto profile_events = profiler.GetProfileEvents();
    for (size_t i = 0; i < profile_events.size(); i++) {
      auto op_index = profile_events[i]->event_metadata;
      const auto node_and_registration =
          interpreter->node_and_registration(int(op_index));
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], op_index, registration);
    }*/
  }

  return EXIT_SUCCESS;
}
