#!/bin/bash -v
python ../tensorflow/tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir mnist_frozen.pb --log_dir /tmp/mnist
tensorboard --logdir /tmp/mnist &
firefox http://localhost:6007/
