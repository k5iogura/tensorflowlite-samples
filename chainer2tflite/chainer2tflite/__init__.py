import logging

from chainer2tflite.export import export  # NOQA

def get_module_logger(modname):
    logger = logging.getLogger(modname)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('chainer2tflite')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
