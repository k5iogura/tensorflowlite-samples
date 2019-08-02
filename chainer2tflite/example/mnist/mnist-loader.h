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
#ifndef EXAMPLE_MNIST_LOADER_H_
#define EXAMPLE_MNIST_LOADER_H_

#include <string>
#include <vector>

namespace example {

///
/// Loads MNIST data.
///
/// @param[in] mnist_label_filename Label filename(t10k-labels-idx1-ubyte)
/// @param[in] mnist_image_filename Image filename(t10k-images-idx3-ubyte)
/// @param[out] images MNIST image data
/// @param[int] labels MNIST labels
///
/// Assume label file and image file has 10000 entries.
///
/// @return true upon success.
///
bool LoadMNISTData(const std::string &mnist_label_filename,
                   const std::string &mnist_image_filename,
                   std::vector<uint8_t> *images, std::vector<int> *labels);

}  // namespace example

#endif // EXAMPLE_MNIST_LOADER_H_
