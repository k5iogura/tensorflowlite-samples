import os

import chainer
import numpy as np

from chainer2tflite import export


def export_testcase(model, args, out_dir, **kwargs):
    """Export model and output tensors of the model in protobuf format.

    This function saves the model with the name "model.tflite".

    Args:
        model (~chainer.Chain): The model object.
        args (list): The arguments which are given to the model
            directly. Unlike `export` function, only `list` type is accepted.
        out_dir (str): The directory name used for saving the input and output.
        **kwargs (dict): keyword arguments for ``chainer2tflite.export``.
    """
    os.makedirs(out_dir, exist_ok=True)

    inputs, outputs = export(
        model, args, filename=os.path.join(out_dir, 'model.tflite'))

    test_data_dir = os.path.join(out_dir, 'test_data_set_0')
    os.makedirs(test_data_dir, exist_ok=True)


    # Save as npy format.

    assert isinstance(inputs, list)
    for i, var in enumerate(inputs):
        npy_name = os.path.join(test_data_dir, 'input_{}.npy'.format(i))
        array = var.array
        np.save(npy_name, array)

    assert isinstance(outputs, list)
    for i, var in enumerate(outputs):
        npy_name = os.path.join(test_data_dir, 'output_{}.npy'.format(i))
        array = var.array
        np.save(npy_name, array)
