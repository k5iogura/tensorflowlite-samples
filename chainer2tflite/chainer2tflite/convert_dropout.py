import numpy as np

from . import serialize_ops


# TODO(LTE): Support mask
# TODO(LTE): Support non float32 type.
def ConvertDropout(converter, serializer, inp, layer_name, parent_layer_name, dropout_ratio):
    """Decompose Dropout function into Constant tensorstant Tensor.

    Args:
        converter(TensorFlowLiteConverter): converter instance
        serializer(TensorFlowLiteSerializer): tflite serializer instance
        inp(chainer.Variable): Input variable
        layer_name(string): Unique layer name
        parent_layer_name(string): Parent layer name
        dropout_ratio(float): Dropout ratio([0.0, 1.0))

    Returns:

    """

    # Generate a tensor filled with uniform random values..
    # This means tensor value is constant and no random number generation in runtime.
    # This behavior is different from Chainer's one.
    # Chainer uses numpy.random without seed, thus it will generate random tensor per run.

    # dropout in Chainer:
    #   scale = 1.0 / (1.0 - dropout_ratio)
    #   flag = rnd >= dropout_ratio
    #   mask = scale * flag
    #   y = x * mask

    #
    # dropout in TensorFlow(Lite) r1.13:
    #   keep_prob = 1 - ratio
    #
    #   [keep_prb, 1.0 + keep_prob)
    #   random_tensor = keep_prob
    #   random_tensor += random_uniform()
    #
    #   0. if [keep_prob, 1.0) and 1.0 if [1.0, 1.0 + keep_prob)
    #   binary_tensor = floor(random_tensor)
    #   ret = divide(x, keep_prob) * binary_tensor
    #
    # We go with TensorFlow way.

    # input
    if inp.name in converter.input_names:
        # Placeholder input
        input_id = serializer.SerializeTensor(
            inp.name, inp.dtype, inp.shape, None)
        converter.inputs[inp.name] = input_id
    elif parent_layer_name == 'data':
        # Constant
        input_id = serializer.SerializeTensor(
            layer_name + '_input0', inp.data.dtype,
            inp.shape, inp.data)
    else:
        input_id = serializer.FindConnection(
            parent_layer_name)
        # There should have valid connection
        if input_id is None:
            print('{} not found in connections'.format(
                parent_layer_name))
            raise


    keep_prob = 1 - dropout_ratio

    #
    # random_tensor = keep_prob
    #
    # Create 1D tensor which contains tensor shape information.
    shape_array = np.array(inp.shape, dtype=np.int32)
    print('shape_array', shape_array)
    shape_id = serializer.SerializeTensor(layer_name + '_shape', 'int32', [len(inp.shape)], shape_array)

    # Create 0D tensor with constant scalar value.
    constant_value = np.array([keep_prob], dtype=np.float32)
    constant_id = serializer.SerializeTensor(layer_name + '_keep_prob_fill', 'float32', [], constant_value)

    # A tenor filled with `keep_prob` value.
    keep_prob_id = serializer.SerializeTensor(layer_name + '_keep_prob', 'float32', inp.shape, None)

    serialize_ops.SerializeOpFill(serializer, shape_id, constant_id, keep_prob_id)


    #
    # random_tensor += random_uniform()
    #

    # [0.0, 1.0)
    rand_array = np.random.rand(*inp.shape).astype(np.float32)

    rand_constant_id = serializer.SerializeTensor(layer_name + '_randm_uniform', 'float32', inp.shape, rand_array)

    rand_id = serializer.SerializeTensor(layer_name + '_random', 'float32', inp.shape, None)

    serialize_ops.SerializeOpAdd(serializer, keep_prob_id, rand_constant_id, rand_id)

    #
    # binary_tensor = floor(random_tensor)
    #
    binary_id = serializer.SerializeTensor(layer_name + '_binary', 'float32', inp.shape, None)

    serialize_ops.SerializeOpFloor(serializer, rand_id, binary_id)

    #
    # divide(x, keep_prob)
    # TODO(LTE): We can precompute `floor(random_tensor)` since dropout_ratio is a constant value
    #            in inference phase.
    #

    divide_id = serializer.SerializeTensor(layer_name + '_divide', 'float32', inp.shape, None)
    serialize_ops.SerializeOpDiv(serializer, input_id, keep_prob_id, divide_id)

    #
    # divide(x, keep_prob) * binary_tensor
    #
    dropout_id = serializer.SerializeTensor(layer_name + '_dropout', 'float32', inp.shape, None)
    serialize_ops.SerializeOpMul(serializer, divide_id, binary_id, dropout_id)


