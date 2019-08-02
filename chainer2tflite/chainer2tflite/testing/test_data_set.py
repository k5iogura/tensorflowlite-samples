import os
import shutil

import chainer2tflite.export_testcase


TEST_OUT_DIR = 'out'

def gen_test_data_set(model, args, name, input_names,
                      output_names):
    model.xp.random.seed(42)
    test_path = os.path.join(
        TEST_OUT_DIR, 'tflite', name)

    chainer2tflite.export_testcase.export_testcase(
        model, args, test_path,
        input_names=input_names, output_names=output_names)
    return test_path


def clean_test_data_set(name):
    test_path = os.path.join(
        TEST_OUT_DIR, 'tflite', name)

    shutil.rmtree(test_path, ignore_errors=True)

