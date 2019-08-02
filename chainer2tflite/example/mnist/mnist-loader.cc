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

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "mnist-loader.h"

namespace example {

namespace {

static bool IsBigEndian(void) {
  union {
    unsigned int i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 1;
}

static void swap4(uint32_t *val) {
  if (!IsBigEndian()) {
    uint32_t tmp = *val;
    uint8_t *dst = reinterpret_cast<uint8_t *>(val);
    uint8_t *src = reinterpret_cast<uint8_t *>(&tmp);

    dst[0] = src[3];
    dst[1] = src[2];
    dst[2] = src[1];
    dst[3] = src[0];
  }
}

} // namespace

bool LoadMNISTData(const std::string &mnist_label_filename,
                   const std::string &mnist_image_filename,
                   std::vector<uint8_t> *images, std::vector<int> *labels) {
  // NOTE(LTE): Assume data is stored in big endian(e.g. POWER, non-Intel CPU)

  // label
  {
    std::ifstream ifs(mnist_label_filename.c_str());
    if (!ifs) {
      std::cerr << "Failed to load label file : " << mnist_label_filename
                << std::endl;
      return false;
    }

    uint32_t magic;

    ifs.read(reinterpret_cast<char *>(&magic), 4);
    swap4(&magic);

    uint8_t data_type = *(reinterpret_cast<uint8_t *>(&magic) + 1);
    uint8_t dim = *(reinterpret_cast<uint8_t *>(&magic) + 0);

    assert(data_type == 8);
    assert(dim == 1);

    uint32_t size = 0;
    ifs.read(reinterpret_cast<char *>(&size), 4);
    swap4(&size);

    assert(size == 10000);

    std::vector<char> buf(size);
    ifs.read(buf.data(), size);

    // byte -> int
    labels->resize(size);
    for (size_t i = 0; i < size_t(size); i++) {
      (*labels)[i] = int(buf[i]);
    }
  }

  // image
  {
    std::ifstream ifs(mnist_image_filename.c_str());
    if (!ifs) {
      std::cerr << "Failed to load image file : " << mnist_image_filename
                << std::endl;
      return false;
    }

    uint32_t magic;

    ifs.read(reinterpret_cast<char *>(&magic), 4);

    uint8_t data_type = *(reinterpret_cast<uint8_t *>(&magic) + 2);
    uint8_t dim = *(reinterpret_cast<uint8_t *>(&magic) + 3);

    assert(data_type == 8);
    assert(dim == 3);

    uint32_t size[3];
    ifs.read(reinterpret_cast<char *>(size), 4 * 3);
    swap4(&size[0]);
    swap4(&size[1]);
    swap4(&size[2]);

    assert(size[0] == 10000);
    assert(size[1] == 28);
    assert(size[2] == 28);

    size_t length = size[0] * size[1] * size[2];
    images->resize(length);
    ifs.read(reinterpret_cast<char *>(images->data()), std::streamsize(length));
  }

  return true;
}

} // namespace example
