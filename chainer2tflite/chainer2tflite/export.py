# -*- coding: utf-8 -*-
import collections
import heapq

# logging
import logging
from logging import getLogger

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import function
from chainer import function_node

from chainer import serializers
from chainer import Link, Chain, ChainList

import flatbuffers

# FIXME(LTE): Find better way of importing tflite
from . import tflite
from .tflite import Buffer
from .tflite import TensorType
from .tflite import Tensor
from .tflite import Model
from .tflite import OperatorCode
from .tflite import SubGraph

from . import serialize_ops

from . import convert_dropout

# default log format
default_fmt = logging.Formatter(
    '[%(asctime)s] %(levelname)s '
    '(%(process)d) %(name)s : %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')

# set up handler
try:
    # Rainbow Logging
    import sys
    from rainbow_logging_handler import RainbowLoggingHandler
    default_handler = RainbowLoggingHandler(sys.stdout)
except Exception:
    default_handler = logging.StreamHandler()

default_handler.setFormatter(default_fmt)
default_handler.setLevel(logging.INFO)

logger = getLogger(__name__)
logger.addHandler(default_handler)

_function_types = (function.Function, function_node.FunctionNode)

#
# TODO(LTE):
#
# * [ ] Consider endianness.
# * [ ] Label input/output tensor before serializing Tensor
#

# Based on Chainre's caffe exporter

# ===========================================================================
#
# Copyright (c) 2015 Preferred Infrastructure, Inc.
#
# Copyright (c) 2015 Preferred Networks, Inc.
#
# See LICENSE of Chainer for details.


def _dump_graph(outputs):
    fan_out = collections.defaultdict(int)
    cand_funcs = []

    def add_cand_to_check(cands):
        for cand in cands:
            x = cand.creator
            if x is None:
                continue
            if x not in fan_out:
                # `len(fan_out)` is in order to avoid comparing `x`
                heapq.heappush(cand_funcs, (-x.rank, len(fan_out), x))
            fan_out[x] += 1

    add_cand_to_check(outputs)
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        assert isinstance(func, _function_types)
        add_cand_to_check(func.inputs)

    ret = []
    cand_funcs = []
    seen_set = set()

    def add_cand(cands):
        cands = [cand.creator for cand in cands if cand.creator is not None]
        for x in cands:
            if x in seen_set:
                continue
            order = 1
            if fan_out[x] == 1 and len(cands) == 1:
                order = -len(seen_set)
            # Negate since heapq is min-heap
            # `len(seen_set)` is in order to avoid comparing `x`
            heapq.heappush(cand_funcs, (order, -x.rank, -len(seen_set), x))
            seen_set.add(x)

    add_cand(outputs)
    while cand_funcs:
        _, _, _, func = heapq.heappop(cand_funcs)
        ret.append(func)
        add_cand(func.inputs)

    return ret[::-1]


# ===========================================================================


class TensorFlowLiteSerializer:
    def __init__(self):
        self.builder = flatbuffers.Builder(0)

        self.buffers = []  # Records Buffer pos

        # 0th buffer must have empty buffer
        self.SerializeBuffer(None)

        self.tensors = []  # Records Tensor pos

        # List of builtin opcodes.
        # This information is required for serializing Model
        self.builtin_opcodes = []

        # List of network operators
        self.operators = []

        # The number of tensor ids(for inputs/outputs in subgraph)
        self.num_tensor_ids = 0

        # variable id <-> tensor id map
        self.variable_id_to_tensor_id = {}

        # name <-> tensor id map
        self.name_to_tensor_id = {}

        self.logger = logger

    # Lookup variable_id and return corresponding tensor id if found.
    def FindTensorIdByVariableId(self, variable_id):
        if variable_id in self.variable_id_to_tensor_id:
            return self.variable_id_to_tensor_id[variable_id]

        return None

    # Register tensor id with variable id
    def RegisterTensorIdWithVariableId(self, variable_id, tensor_id):
        if variable_id in self.variable_id_to_tensor_id:
            logger.fatal(
                'VariableId({}) is already registered.'.format(variable_id))
            raise

        self.variable_id_to_tensor_id[variable_id] = tensor_id

    # Lookup name and return corresponding tensor id if found.
    def FindTensorIdByName(self, name):
        if name in self.name_to_tensor_id:
            return self.name_to_tensor_id[name]

        return None

    # Register variable with (unique) name.
    def RegisterTensorIdWithName(self, name, tensor_id):
        if name in self.name_to_tensor_id:
            logger.fatal('name({}) is already registered.'.format(name))
            raise

        self.name_to_tensor_id[name] = tensor_id

    def EmitTensorId(self):

        # Assign tensor id  based on number of tensor ids
        tensor_id = self.num_tensor_ids

        self.num_tensor_ids = self.num_tensor_ids + 1

        return tensor_id

    def RegisterBuiltinOpcode(self, opcode):
        """Register tflite's Builtin opcode

        Args:

            opcode (tflite enum) : BuiltinOpcode enum

        Returns:

            An arary index to registered opcode
        """

        if opcode in self.builtin_opcodes:
            # opcode is already registered
            return self.builtin_opcodes.index(opcode)

        # Add opcode
        self.builtin_opcodes.append(opcode)

        return self.builtin_opcodes.index(opcode)

    def SerializeBuffer(self, data):
        """
            data : bytearray or None(empty)
        """

        data_len = 0
        if data is not None:
            data_len = len(data)

        if data_len > 0:
            # Serialize tensor data: [uint8]
            # https://github.com/google/flatbuffers/issues/4668
            buffer_start = tflite.Buffer.BufferStartDataVector(
                self.builder, data_len)

            # We need to seek the header to correct place before writing into
            # Bytes array
            self.builder.head = self.builder.head - data_len
            self.builder.Bytes[self.builder.head:(self.builder.head +
                                                  data_len)] = data

            tf_data = self.builder.EndVector(data_len)

        tflite.Buffer.BufferStart(self.builder)
        if data_len > 0:
            tflite.Buffer.BufferAddData(self.builder, tf_data)
        tf_buffer = tflite.Buffer.BufferEnd(self.builder)

        buffer_id = len(self.buffers)
        self.buffers.append(tf_buffer)

        return (buffer_id, tf_buffer)

    def SerializeTensor(self, name, dtype, shape, data):
        """Serialize Tensor.

        Currently we only support Tensor with float32 format.

        Args:
            name (string): (Unique) name of Tensor.
            dtype (numpy.dtype): Tensor data type.
            shape ([int]): Tensor shape information.
            data (chainer.Variable or numpy.ndarray): Tensor data.
                Create empty tensor when `data` is None

        Returns:
            tensor id(int)
        """

        # TODO(LTE): Support other types
        tf_type = None
        if dtype == 'float32':
            tf_type = tflite.TensorType.TensorType.FLOAT32
        elif dtype == 'int32':
            tf_type = tflite.TensorType.TensorType.INT32
        else:
            print('Unsupported data type :', dtype)
            raise

        # Serialize Tensor data: [ubyte]
        if data is not None:
            # Sequentially assign buffer id.
            buffer_id = len(self.buffers)

            tensor_values = data.flatten()  # numpy.ndarray
            self.SerializeBuffer(tensor_values.tobytes())
        else:
            # point to empty buffer
            buffer_id = 0

        # Serialize shape: [int32]
        tflite.Tensor.TensorStartShapeVector(self.builder, len(shape))

        for i in reversed(range(len(shape))):
            self.builder.PrependInt32(shape[i])
        tf_shape = self.builder.EndVector(len(shape))

        tf_name = self.builder.CreateString(name)

        # Buld Tensor table
        tflite.Tensor.TensorStart(self.builder)
        tflite.Tensor.TensorAddName(self.builder, tf_name)
        tflite.Tensor.TensorAddShape(self.builder, tf_shape)
        tflite.Tensor.TensorAddType(self.builder, tf_type)
        tflite.Tensor.TensorAddBuffer(self.builder, buffer_id)
        tf_tensor = tflite.Tensor.TensorEnd(self.builder)

        tensor_id = self.EmitTensorId()
        self.tensors.append(tf_tensor)

        logger.info("Tensor[{}] name = {}, shape = {}".format(tensor_id, name, shape))

        return tensor_id

    def SerializeSubGraph(self, inputs, outputs):
        """Serialize SubGraph.

        Args:
            inputs ([int]) : List of input ids.
            outputs ([int]) : List of output ids.
        """

        logger.info("Num inputs = %d", len(inputs))
        logger.info("  %s", inputs)
        logger.info("Num outputs = {}".format(len(outputs)))
        logger.info("  %s", outputs)
        logger.info("Num tensors = {}".format(len(self.tensors)))
        logger.info("Num operators = {}".format(len(self.operators)))

        # [Inputs]
        tflite.SubGraph.SubGraphStartInputsVector(self.builder, len(inputs))
        for i in reversed(inputs):
            self.builder.PrependInt32(i)
        tf_inputs = self.builder.EndVector(len(inputs))

        # [Outputs]
        tflite.SubGraph.SubGraphStartOutputsVector(self.builder, len(outputs))
        for o in reversed(outputs):
            self.builder.PrependInt32(o)
        tf_outputs = self.builder.EndVector(len(outputs))

        # [Operators]
        tflite.SubGraph.SubGraphStartOperatorsVector(self.builder,
                                                     len(self.operators))
        for o in reversed(self.operators):
            self.builder.PrependUOffsetTRelative(o)
        tf_operators = self.builder.EndVector(len(self.operators))

        # [Tensors]
        logger.info('self.tensors = %d', len(self.tensors))
        tflite.SubGraph.SubGraphStartTensorsVector(self.builder,
                                                   len(self.tensors))
        for tensor_pos in reversed(self.tensors):
            logger.info('tensor_pos = %d', tensor_pos)
            self.builder.PrependUOffsetTRelative(tensor_pos)
        tf_tensors = self.builder.EndVector(len(self.tensors))

        # TODO(syoyo): subgraph name
        tf_name = self.builder.CreateString("Nyaan")

        tflite.SubGraph.SubGraphStart(self.builder)
        tflite.SubGraph.SubGraphAddInputs(self.builder, tf_inputs)
        tflite.SubGraph.SubGraphAddOutputs(self.builder, tf_outputs)
        tflite.SubGraph.SubGraphAddOperators(self.builder, tf_operators)
        tflite.SubGraph.SubGraphAddTensors(self.builder, tf_tensors)
        tflite.SubGraph.SubGraphAddName(self.builder, tf_name)

        subgraph = tflite.SubGraph.SubGraphEnd(self.builder)

        return subgraph

    def SerializeModel(self, subgraph):

        # [Buffers]
        tflite.Model.ModelStartBuffersVector(self.builder, len(self.buffers))

        for i in reversed(range(len(self.buffers))):
            self.builder.PrependUOffsetTRelative(self.buffers[i])

        tf_buffers = self.builder.EndVector(len(self.buffers))

        # [Subgraphs]
        # Currently we only support 1 subgraphs in a model.
        tflite.Model.ModelStartSubgraphsVector(self.builder, 1)
        self.builder.PrependUOffsetTRelative(subgraph)
        tf_subgraphs = self.builder.EndVector(1)

        # [OperatorCodes]
        tf_opcodes = []
        for k in self.builtin_opcodes:
            tflite.OperatorCode.OperatorCodeStart(self.builder)
            logger.info('code = %d', k)
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(self.builder, k)
            tf_opcode = tflite.OperatorCode.OperatorCodeEnd(self.builder)
            logger.info('tf_opcode = %d', tf_opcode)

            tf_opcodes.append(tf_opcode)

        tflite.Model.ModelStartOperatorCodesVector(self.builder,
                                                   len(tf_opcodes))
        for i in reversed(range(len(tf_opcodes))):
            self.builder.PrependUOffsetTRelative(tf_opcodes[i])
        opcodes = self.builder.EndVector(len(tf_opcodes))

        tflite.Model.ModelStart(self.builder)

        # version must be 3(or higher?)
        tflite.Model.ModelAddVersion(self.builder, 3)
        tflite.Model.ModelAddSubgraphs(self.builder, tf_subgraphs)
        tflite.Model.ModelAddBuffers(self.builder, tf_buffers)
        tflite.Model.ModelAddOperatorCodes(self.builder, opcodes)
        model = tflite.Model.ModelEnd(self.builder)

        return model

    def GetOutput(self, rootTable):

        # file_identifier is missing in python binding
        # (At least flatbuffers ~1.11).
        # https://github.com/google/flatbuffers/issues/4814
        #
        # `file_identifier` is required when reading it in TensorFlow Lite C++.
        # Manually add `file_identifier` here

        file_identifier = 'TFL3'

        prepSize = flatbuffers.number_types.UOffsetTFlags.bytewidth + len(
            file_identifier)  # = 8
        self.builder.Prep(self.builder.minalign, prepSize)

        b = bytes(file_identifier, encoding='utf-8')
        for i in reversed(b):
            self.builder.PrependByte(i)

        self.builder.Finish(rootTable)

        return self.builder.Output()


class TensorFlowLiteConverter(object):

    debug = False

    def __init__(self, tflitemodel=None):
        self.tflitemodel = tflitemodel
        # key:string, val:dict(key: func, val: index)
        self.naming_map = collections.defaultdict(dict)

        # Placeholder input tensor id
        # Will be found during `dump_function_object`
        self.inputs = {}

        # List of input names
        self.input_names = []

    def _fold_transpose_and_conv2d(self, funcs):
        """
        Detect such a sequence of functions then remove Transpose.

            chainer.functions.array.transpose.Transpose(W)
            chainer.functions.connection.convolution_2d.Convolution2DFunction(W)

        Args:
            funcs : List of functions

        Returns:
            List of functions where Transpose func has been removed.
        """

        if len(funcs) < 2:
            return funcs

        out_funcs = []

        # TODO(LTE): Refactor
        i = 1
        for j in range(1, len(funcs), 2):
            i = j

            W_transpose = False

            if (funcs[i - 1].label == 'Transpose') and (funcs[i - 3].axes == (
                    1, 0, 2, 3)):
                print("bora", funcs[i - 1].inputs[0].data)
                if funcs[i].label == 'Convolution2DFunction':
                    print("bingo")
                    # TODO(LTE): Ensure input to `Convolution2DFunction` is the output of `Transpose`.
                    W_transpose = True

            if W_transpose:
                out_funcs.append(funcs[i])
            else:
                out_funcs.append(funcs[i - 1])
                out_funcs.append(funcs[i])

        # remainder
        if (i + 1) < len(funcs):
            for j in range(i + 1, len(funcs)):
                out_funcs.append(funcs[j])

        return out_funcs

    def _fold_depthwise_conv2d(self, funcs):
        """
        DepthwiseConvolution2D is decomposed into the following 3 functions.
        Detect such a functions and remove Transpose and Reshape.

            chainer.functions.array.transpose.Transpose
            chainer.functions.array.reshape.Reshape
            chainer.functions.connection.convolution_2d.Convolution2DFunction


        Args:
            funcs : List of functions


        Returns:
            List of functions where Transpose And Reshape func has been removed.
        """

        if len(funcs) < 3:
            return funcs

        out_funcs = []

        # TODO(LTE): Refactor
        i = 2
        for j in range(2, len(funcs), 3):

            i = j

            depthwise_conv2d = False

            # Use `id`(pointer address) to identitify the input of successor
            # FunctionNode are connected to the output of predecessor
            # FunctionNode.
            # Note: output is a weakref object.
            if (funcs[i - 2].label == 'Transpose') and (funcs[i - 2].axes == (
                    1, 0, 2, 3)):
                if funcs[i - 1].label == 'Reshape':
                    # Ensure Transpose.output == Reshape.input
                    if funcs[i - 1].inputs[0] != funcs[i - 2].outputs[0]():
                        break

                    shape = funcs[i - 1].outputs[0]().shape

                    if (len(shape) == 4) and (shape[1] == 1):
                        if funcs[i].label == 'Convolution2DFunction':

                            # Ensure Reshape.output == Convolution2DFunction.W(input[1])
                            if funcs[i].inputs[1] != funcs[i - 1].outputs[0]():
                                break

                            # Bingo!
                            depthwise_conv2d = True

            if depthwise_conv2d:
                out_funcs.append(funcs[i])
            else:
                out_funcs.append(funcs[i - 2])
                out_funcs.append(funcs[i - 1])
                out_funcs.append(funcs[i])

        # remainder
        if (i + 1) < len(funcs):
            for j in range(i + 1, len(funcs)):
                out_funcs.append(funcs[j])

        return out_funcs

    def _get_layer_name(self, layer):
        """Generate layer name like "Convolution2DFunction-10-2".

        The first number means rank of the layer (depth from the top),
        and the second number is for preventing duplication
        (different layer objects can have same rank)

        Args:
            layer (~chainer.Function_node): Function object
        Returns:
            str: A string to be used for the ``name`` field of the graph
                in the exported Caffe model.

        """
        label = '{}-{}'.format(layer.label, layer.rank)
        d = self.naming_map[label]
        if layer not in d.keys():
            d[layer] = len(d) + 1
        return '{}-{}'.format(label, d[layer])

    def _get_parent_name(self, parent_):
        if parent_ is None:
            return 'data'
        return self._get_layer_name(parent_)

    def _insertOpTransposeTensor(self, tf_serializer, input_id, in_shape,
                                 in_dtype, perm, output_shape, output_name,
                                 tensor_format_prefix):
        """Insert Transpose op to convert Tensor format(Usually for NCHW <-> NHWC conversion)

        Args:
            input_id(int): Input Tensor id
            perm([int]): Array of axes for permutation(e.g. [0, 2, 3, 1])
            tensor_format_prefix(str) : Prefix to add tensor name. Usually '_nchw' or '_nhwc'
        """

        # Must be 4D shape
        assert len(perm) == 4
        assert len(in_shape) == 4
        assert len(output_shape) == 4

        assert (tensor_format_prefix == '_nchw') or (
            tensor_format_prefix == '_nhwc')

        # perm
        perm = np.array(perm).astype(np.int32)

        perm_id = tf_serializer.SerializeTensor(output_name + '_perm',
                                                perm.dtype, perm.shape, perm)

        # output Tensor
        output_id = tf_serializer.SerializeTensor(output_name, in_dtype,
                                                  output_shape, None)
        tf_serializer.RegisterTensorIdWithName(output_name, output_id)

        serialize_ops.SerializeOpTranspose(tf_serializer, input_id, perm_id,
                                           output_id)

        return output_id

    def _insertOpPad(self, tf_serializer, input_id, in_shape, in_dtype,
                     in_data, output_shape, pad_bw, pad_value, layer_name,
                     tensor_format_prefix):
        """Insert Pad op for converting pooling op with padding > 0 in Chainer to tflite.
           Assume input is existing Tensor(not a constant).
        """

        # create a tensor for padding.

        # tflite = 2D tensor with [begin, end]x ndim. For example:
        # [[pad0_b, pad0_e],
        #  [pad1_b, pad1_e],
        #  [pad2_b, pad2_e],
        #  [pad3_b, pad3_e]]
        print('func_pad = ', pad_bw)
        padding_values = []
        for ps in pad_bw:
            if isinstance(ps, np.ndarray):
                padding_values.append([ps[0], ps[1]])
            elif isinstance(ps, list) or isinstance(ps, tuple):
                padding_values.append([ps[0], ps[1]])
            else:
                # Should be int-typed value
                assert type(ps) == int
                padding_values.append([ps, ps])

        print('padding values = ', padding_values)

        # paddig tensor must have same array length for the first axis with input tensor
        padding = np.array(padding_values, np.int32)

        print('padding.shape = ', padding.shape)
        print('padding = ', padding)
        padding_id = tf_serializer.SerializeTensor(layer_name + '_padding',
                                                   padding.dtype, padding.shape,
                                                   padding)

        constant_id = -1
        # create a constant value tensor used for padding value.
        constant_value = np.array([pad_value], np.float32)
        constant_id = tf_serializer.SerializeTensor(
            layer_name + '_constant_value', constant_value.dtype,
            constant_value.shape, constant_value)

        # output
        output_id = tf_serializer.SerializeTensor(layer_name + '_0', in_dtype,
                                                  output_shape, None)
        tf_serializer.RegisterTensorIdWithName(layer_name + '_0', output_id)

        serialize_ops.SerializeOpPad(tf_serializer, input_id, output_id,
                                     padding_id, constant_id)

        return output_id

    def dump_function_object(self, func, tf_serializer):

        assert isinstance(func, _function_types)
        layer_name = self._get_layer_name(func)

        parent_layer_names = [
            self._get_parent_name(input_.creator) for input_ in func.inputs
        ]

        layer = None

        for input_ in func.inputs:
            logger.info('input name = %s', input_.name)

        logger.info('label = %s', func.label)
        logger.info('len(inputs) = %d', len(func.inputs))
        logger.info('top = %s', layer_name)
        logger.info('parent_layer_names = %s', parent_layer_names)

        # NOTE(LTE): `func.outputs` is a weakref.
        # So use '()' to deref it when you access `func.outputs`

        if func.label == 'LinearFunction':
            #
            # TODO(syoyo): Convert LinearFunction + ReLU to
            # FULLY_CONNECTED with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('Linear in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            b = None
            if len(func.inputs) == 2:
                inp, W = func.inputs
            else:  # guess 3
                inp, W, b = func.inputs

            # input
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # W
            W_id = tf_serializer.SerializeTensor(parent_layer_names[1],
                                                 W.dtype, W.shape, W.data)

            # b
            b_id = -1  # -1 = optional
            if b is not None:
                if b.data is not None:
                    b_id = tf_serializer.SerializeTensor(
                        parent_layer_names[2], b.dtype, b.shape, b.data)

            # output
            _output = func.outputs[0]
            logger.info("Linear output.id = {}".format(id(_output())))
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      inp.dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            activation_function = 'NONE'
            serialize_ops.SerializeOpFullyConnected(tf_serializer,
                                                    activation_function,
                                                    input_id, output_id, W_id,
                                                    b_id)

        elif func.label == 'Convolution2DFunction':
            for _input in func.inputs:
                logger.info('Convolution2DFunction in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) >= 2

            inp = func.inputs[0]
            in_id = id(inp)
            in_name = inp.name
            in_dtype = inp.dtype
            in_shape = inp.shape
            in_data = inp.data

            input_is_variable = False

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.shape, in_dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

                in_shape = output_shape

                input_is_variable = True

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, in_shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])
                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, in_shape, in_dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                    input_is_variable = True

                else:
                    # Input is constant. There is nothing to do here.
                    pass

                # To NHWC
                in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                            inp.shape[1])

                if in_data is not None:
                    in_data = np.transpose(inp.data, (0, 2, 3, 1))

            #
            # Hereafter, input tensor has NHWC shape.
            #

            num_groups = func.groups

            depthwise = False
            if num_groups > 1:
                # Assume depthwise convolution
                depthwise = True

            # filter
            # shape = [outC, inC, kh, kw]
            filt = func.inputs[1]
            filt_dtype = filt.dtype
            filt_data = filt.data
            print('Chainer filt_data = ', filt.data)

            out_channels = filt.shape[0]
            in_channels = filt.shape[1]

            if depthwise:

                # In Chainer, DepthwiseConvolution2D is transformed into
                #
                # multiplier, in_channels, kh, kw = W.shape
                #
                # W = transpose(W, (1, 0, 2, 3))
                # W = reshape(W, (multiplier * in_channels, 1, kh, kw))
                # convolution_2d(x, W, b, stride, pad, groups=in_channels)

                multiplier = int(filt.shape[0] /
                                 num_groups)  # Should be integer dividable

                # TFLite expects [1, kh, kw, multiplier * in_channels]
                # where out_channels == mult * in_channels

                # [mult * num_groups, 1, kh, kw] -> [1, kh, kw, mult * in_channels]
                tf_filt_data = np.transpose(filt_data, (1, 2, 3, 0))

                assert filt.shape[1] == 1

                tf_filt_shape = (1, filt.shape[2], filt.shape[3],
                                 filt.shape[0])

            else:
                # tflite CONV_2D filt = [outC, kh, kw, inC]
                # Apply [outC, inC, kh, kw] -> [outC, kh, kw, inC] conversion
                tf_filt_shape = (out_channels, filt.shape[2], filt.shape[3],
                                 in_channels)

                tf_filt_data = np.transpose(filt.data, (0, 2, 3, 1))

            parent_layer_name = parent_layer_names[0]

            pad_required = True
            tf_padding_mode = 'VALID'
            print('Conv2d.depthwise = ', depthwise)
            print('func, pad = ', func.ph, func.pw)
            print('Chainer Conv2d.filt_shape = ', filt.shape)
            print('tflite CONV_2D.filt_shape = ', tf_filt_shape)
            print('tflite CONV_2D.filt_data = ', filt_data)

            # TODO(LTE): Support uneven padding
            assert func.ph == func.pw

            if (func.ph == 0) and (func.pw == 0):
                pad_required = False
            elif (func.ph == 1) and (func.pw == 1):
                if (tf_filt_shape[1] == 3) and (tf_filt_shape[2] == 3):
                    # zero padding with padding width 1.
                    # We can use 'SAME' padding mode in tflite.
                    pad_required = False
                    tf_padding_mode = 'SAME'

            if pad_required:
                # Insert `Pad` op

                _layer_name = layer_name + '_pad'
                pad_bw = []

                if len(in_shape) == 4:
                    # NHWC
                    _output_shape = [
                        sum(x) for x in zip(in_shape,
                                            [0, 2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([0, 0])
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])
                    pad_bw.append([0, 0])
                elif len(in_shape) == 3:
                    # HWC
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])
                    pad_bw.append([0, 0])
                else:
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])

                logger.info('pad output shape = {}'.format(_output_shape))

                # padding constant value for conv2 is 0.0.
                pad_value = 0.0

                pad_id = self._insertOpPad(tf_serializer, input_id, in_shape,
                                           in_dtype, in_data, _output_shape,
                                           pad_bw, pad_value, _layer_name,
                                           '_nhwc')

                # Overwrite input tensor id
                input_id = pad_id

            else:
                if not input_is_variable:
                    if parent_layer_names[0] == 'data':
                        input_id = tf_serializer.SerializeTensor(
                            layer_name + '_input0', inp.dtype, in_shape,
                            in_data)
                    else:
                        raise

            # filter
            filter_id = tf_serializer.SerializeTensor(parent_layer_names[1],
                                                      filt_dtype,
                                                      tf_filt_shape,
                                                      tf_filt_data)

            # bias
            b = None
            if len(func.inputs) > 2:
                b = func.inputs[2]

            # Even though bias is optional in tflite,
            # it looks tflite runtime expects a valid bias tensor.
            # thus create zero valud bias tensor if required.
            bias_id = -1
            if (b is not None) and (b.data is not None):
                bias_id = tf_serializer.SerializeTensor(
                    parent_layer_names[2], b.dtype, b.shape, b.data)
            else:
                # Bias is 1D tensor with size `outC`
                bias_shape = [out_channels]
                bias_data = np.zeros(bias_shape, dtype=inp.dtype)

                bias_id = tf_serializer.SerializeTensor(
                    layer_name + '_bias', inp.dtype, bias_shape, bias_data)

            # output
            _output = func.outputs[0]
            print("Chainer output.shape = ", _output().shape)

            output_shape = _output().shape

            # Chainer NCHW -> tflite NHWC
            output_shape = (_output().shape[0], _output().shape[2],
                            _output().shape[3], _output().shape[1])

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name, in_dtype,
                                                      output_shape, None)

            tf_serializer.RegisterTensorIdWithName(output_name, output_id)

            # options
            activation_function = 'NONE'

            stride = [func.sx, func.sy]

            # W, H
            dilations = [1, 1]

            if hasattr(func, 'dy') and hasattr(func, 'dx'):
                dilations[0] = func.dx
                dilations[1] = func.dy

            print('padding mode', tf_padding_mode)
            print('stride', stride)
            print('dilations', dilations)

            assert dilations[0] >= 1 and dilations[1] >= 1

            if depthwise:
                print('multiplier', multiplier)
                print('output shape', output_shape)

                serialize_ops.SerializeDepthwiseConv2D(tf_serializer, input_id,
                                                       filter_id, bias_id,
                                                       output_id,
                                                       activation_function,
                                                       tf_padding_mode, stride,
                                                       dilations, multiplier)

            else:

                serialize_ops.SerializeConv2D(tf_serializer, input_id,
                                              filter_id, bias_id, output_id,
                                              activation_function,
                                              tf_padding_mode, stride,
                                              dilations)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, in_dtype, [0, 3, 1, 2],
                reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'AveragePooling2D':
            #
            # TODO(syoyo): Convert AveragePooling2D + ReLU to
            # AVERAGE_POOL_2D with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('AveragePooling2D in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) == 1

            inp = func.inputs[0]
            in_name = inp.name
            in_id = id(inp)
            in_dtype = inp.dtype
            in_shape = inp.shape
            in_data = inp.data

            # Assume 4D shape
            assert len(in_shape) == 4

            format_prefix = '_nhwc'

            input_is_variable = False

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.shape, in_dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

                in_shape = output_shape

                input_is_variable = True

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, in_shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])
                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, in_shape, in_dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                    input_is_variable = True

                else:
                    # Input is constant. There is nothing to do here.
                    pass

                # To NHWC
                in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                            inp.shape[1])

                if in_data is not None:
                    in_data = np.transpose(inp.data, (0, 2, 3, 1))

            #
            # Hereafter, input tensor has NHWC shape.
            #

            parent_layer_name = parent_layer_names[0]

            pad_required = True
            tf_padding_mode = 'VALID'

            # TODO(LTE): Support uneven padding
            assert func.ph == func.pw

            if (func.ph == 0) and (func.pw == 0):
                pad_required = False

            if pad_required:
                assert input_is_variable is True

                # Insert `Pad` op

                _layer_name = in_name + '_pad'
                pad_bw = []

                if len(in_shape) == 4:
                    # NHWC
                    _output_shape = [
                        sum(x) for x in zip(in_shape,
                                            [0, 2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([0, 0])
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])
                    pad_bw.append([0, 0])
                elif len(in_shape) == 3:
                    # HWC
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])
                    pad_bw.append([0, 0])
                else:
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph, 2 * func.pw, 0])
                    ]
                    pad_bw.append([func.ph, func.ph])
                    pad_bw.append([func.pw, func.pw])

                logger.info('pad output shape = {}'.format(_output_shape))

                # constant value for ave pooling is 0.0
                pad_value = 0.0

                pad_id = self._insertOpPad(tf_serializer, input_id, in_shape,
                                           in_dtype, in_data, _output_shape,
                                           pad_bw, pad_value, _layer_name,
                                           '_nhwc')

                # Replace input_id with pad_id
                input_id = pad_id

                # rewrite parent name to OpPad's name
                parent_layer_name = _layer_name

            else:
                if not input_is_variable:
                    if parent_layer_names[0] == 'data':
                        input_id = tf_serializer.SerializeTensor(
                            layer_name + '_input0', inp.dtype, in_shape,
                            in_data)
                    else:
                        raise

            # output
            _output = func.outputs[0]

            output_shape = _output().shape

            assert len(output_shape) == 4

            # NCHW -> NHWC
            output_shape = (_output().shape[0], _output().shape[2],
                            _output().shape[3], _output().shape[1])

            #print("average_pool_2d.output.shape = {}".format(output_shape))
            logger.info(
                "average_pool_2d.output.shape = {}".format(output_shape))
            output_id = tf_serializer.SerializeTensor(
                layer_name + '_0' + format_prefix, inp.dtype, output_shape,
                None)
            tf_serializer.RegisterTensorIdWithName(
                layer_name + '_0' + format_prefix, output_id)

            # options

            activation_function = 'NONE'

            padding = 'VALID'
            stride = [func.sx, func.sy]
            filter_size = [func.kw, func.kh]
            serialize_ops.SerializeAveragePooling2D(tf_serializer, input_id,
                                                    output_id,
                                                    activation_function,
                                                    padding, stride,
                                                    filter_size)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, in_dtype, [0, 3, 1, 2],
                reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'MaxPooling2D':
            #
            # TODO(syoyo): Convert MaxPooling2D + ReLU to
            # MAX_POOL_2D with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('MaxPooling2D in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) == 1

            logger.info("MaxPooling2D. ph, pw = {}, {}".format(func.ph, func.pw))
            logger.info("MaxPooling2D. sx, sy = {}, {}".format(func.sx, func.sy))
            logger.info("MaxPooling2D. kw, kh = {}, {}".format(func.kw, func.kh))
            logger.info("MaxPooling2D. coverall = {}".format(func.cover_all))

            cover_all = func.cover_all

            inp = func.inputs[0]
            in_name = inp.name
            in_id = id(inp)
            in_dtype = inp.dtype
            in_shape = inp.shape
            in_data = inp.data

            # Get output shape
            # FIXME(LTE): It looks chainer returns wrong output shape in some situation.
            _output = func.outputs[0]
            output_shape = _output().shape
            logger.info('MaxPooling2D output shape = {}'.format(output_shape))

            input_is_variable = False

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.shape, in_dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

                in_shape = output_shape

                input_is_variable = True

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, in_shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])
                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, in_shape, in_dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                    input_is_variable = True

                else:
                    # Input is constant. There is nothing to do here.
                    pass

                # To NHWC
                in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                            inp.shape[1])

                if in_data is not None:
                    in_data = np.transpose(inp.data, (0, 2, 3, 1))

            #
            # Hereafter, input tensor has NHWC shape.
            #

            parent_layer_name = parent_layer_names[0]

            pad_required = True
            tf_padding_mode = 'VALID'

            if (func.ph == 0) and (func.pw == 0) and (cover_all == False):
                pad_required = False

            if pad_required:
                assert input_is_variable is True

                # Insert `Pad` op

                _layer_name = layer_name + '_pad'
                pad_bw = []

                extra_padding = 0
                if cover_all:
                    # FIXME(LTE): Compute correct margin based on shape, padding and stride
                    # https://docs.chainer.org/en/stable/reference/generated/chainer.functions.convolution_2d.html
                    extra_padding = 1

                if len(in_shape) == 4:
                    # NHWC
                    _output_shape = [
                        sum(x) for x in zip(in_shape,
                                            [0, 2 * func.ph + extra_padding, 2 * func.pw + extra_padding, 0])
                    ]
                    pad_bw.append([0, 0])
                    pad_bw.append([func.ph, func.ph + extra_padding])
                    pad_bw.append([func.pw, func.pw + extra_padding])
                    pad_bw.append([0, 0])
                elif len(in_shape) == 3:
                    # HWC
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph + extra_padding, 2 * func.pw + extra_padding, 0])
                    ]
                    pad_bw.append([func.ph, func.ph + extra_padding])
                    pad_bw.append([func.pw, func.pw + extra_padding])
                    pad_bw.append([0, 0])
                else:
                    _output_shape = [
                        sum(x)
                        for x in zip(in_shape, [2 * func.ph + extra_padding, 2 * func.pw + extra_padding, 0])
                    ]
                    pad_bw.append([func.ph, func.ph + extra_padding])
                    pad_bw.append([func.pw, func.pw + extra_padding])

                logger.info('pad output shape = {}'.format(_output_shape))

                # constant value for max pooling is -inf.
                pad_value = -np.inf

                pad_id = self._insertOpPad(tf_serializer, input_id, in_shape,
                                           in_dtype, in_data, _output_shape,
                                           pad_bw, pad_value, _layer_name,
                                           '_nhwc')

                input_id = pad_id

                # rewrite parent name to OpPad's name
                parent_layer_name = _layer_name

            else:
                # input
                if not input_is_variable:
                    if parent_layer_names[0] == 'data':
                        input_id = tf_serializer.SerializeTensor(
                            layer_name + '_input0', inp.dtype, in_shape,
                            in_data)
                    else:
                        raise


            if len(output_shape) == 4:
                # NCHW -> NHWC
                output_shape = (_output().shape[0], _output().shape[2],
                                _output().shape[3], _output().shape[1])

            #print("average_pool_2d.output.shape = {}".format(output_shape))
            logger.info("max_pool_2d.output.shape = {}".format(output_shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      inp.dtype, output_shape,
                                                      None)
            tf_serializer.RegisterTensorIdWithName(layer_name + '_0',
                                                   output_id)

            # options

            activation_function = 'NONE'

            stride = [func.sx, func.sy]
            filter_size = [func.kw, func.kh]
            serialize_ops.SerializeMaxPooling2D(tf_serializer, input_id,
                                                output_id, activation_function,
                                                tf_padding_mode, stride,
                                                filter_size)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, in_dtype, [0, 3, 1, 2],
                reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'Unpooling2D':

            #
            # Map to RESIZE_NEAREST_NEIGHBOR
            #

            for _input in func.inputs:
                logger.info('Unpooling2D in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) == 1

            inp = func.inputs[0]

            in_name = inp.name
            in_id = id(inp)
            in_dtype = inp.dtype
            in_shape = inp.shape
            in_data = inp.data

            assert len(in_shape) == 4

            if (func.sx != func.kw) or (func.sy != func.kh):
                logger.fatal(
                    'stride{} and ksize{} must be same for Unpooling2D in {}(id {})'
                    .format([func.sy, func.sx], [func.kh, func.kw],
                            self._get_parent_name(_input), id(_input)))

            parent_layer_name = parent_layer_names[0]

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, in_dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

                in_shape = output_shape

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, in_shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])

                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, in_shape, inp.dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                elif parent_layer_names[0] == 'data':
                    # Input is constant.

                    # To NHWC
                    in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])

                    if in_data is not None:
                        in_data = np.transpose(inp.data, (0, 2, 3, 1))

                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input0' + format_prefix, inp.dtype,
                        in_shape, in_data)

            # new_shape(1D tensor with 2 elements)
            new_shape_name = layer_name + '_new_shape'
            new_shape_id = tf_serializer.SerializeTensor(
                new_shape_name, 'int32', [2],
                np.array([func.outh, func.outw], dtype=np.int32))

            # output
            _output = func.outputs[0]

            output_shape = _output().shape

            assert len(output_shape) == 4

            # NCHW -> NHWC
            output_shape = (_output().shape[0], _output().shape[2],
                            _output().shape[3], _output().shape[1])

            logger.info("unpooling2d.output.shape = {}".format(output_shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name, inp.dtype,
                                                      output_shape, None)
            tf_serializer.RegisterTensorIdWithName(output_name, output_id)

            serialize_ops.SerializeOpResizeNearestNeighbor(
                tf_serializer, input_id, output_id, new_shape_id)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, in_dtype, [0, 3, 1, 2],
                reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'Transpose':

            assert (len(func.inputs) == 1)

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # perm
            if isinstance(func.axes, int):
                func.axes = [func.axes]

            perm = np.array(func.axes).astype(np.int32)

            perm_id = tf_serializer.SerializeTensor(layer_name + '_perm',
                                                    perm.dtype, perm.shape,
                                                    perm)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpTranspose(tf_serializer, input_id,
                                               perm_id, output_id)

        elif func.label == 'Tile':

            assert (len(func.inputs) == 1)

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # multiples
            if isinstance(func.reps, int):
                func.reps = [func.reps]

            multiples = np.array(func.reps).astype(np.int32)
            multiples_id = tf_serializer.SerializeTensor(
                layer_name + '_multiples', multiples.dtype, multiples.shape,
                multiples)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpTile(tf_serializer, input_id,
                                          multiples_id, output_id)

        elif func.label == 'ExpandDims':

            assert (len(func.inputs) == 1)

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # axis
            axis = np.array(func.axis).astype(np.int32)
            axis_id = tf_serializer.SerializeTensor(layer_name + '_axis',
                                                    axis.dtype, axis.shape,
                                                    axis)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpExpandDims(tf_serializer, input_id,
                                                axis_id, output_id)

        elif func.label == 'Squeeze':

            assert (len(func.inputs) == 1)

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # axis. 1D array
            axis = []
            if func.axis is None:
                for i, s in enumerate(func.inputs[0].shape):
                    if s == 1:
                        axis.append(i)
            else:
                axis = func.axis

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpSqueeze(tf_serializer, input_id,
                                             output_id, axis)

        elif func.label == 'Concat':

            # input
            input_ids = []
            for (i, inp) in enumerate(func.inputs):

                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input{}'.format(i), 'float32',
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            # axis param
            axis = func.axis

            serialize_ops.SerializeOpConcatenation(tf_serializer, input_ids,
                                                   output_id, axis)

        elif func.label == 'SplitAxis':

            inp = func.inputs[0]

            # input
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input{}'.format(0), 'float32', inp.shape,
                    inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[i]))
                    raise

            # output
            output_ids = []
            for (i, outp) in enumerate(func.outputs):

                output_id = tf_serializer.SerializeTensor(
                    layer_name + '_{}'.format(i),
                    outp().dtype,
                    outp().shape, None)

                output_ids.append(input_id)

                tf_serializer.RegisterTensorIdWithVariableId(
                    id(outp()), output_id)

            logger.fatal(
                'SplitAxis is not yet supported(need to implement multiple outputs in chainer2tflite firstly)'
            )
            raise

        elif func.label == 'Space2Depth':

            inp = func.inputs[0]
            in_data = inp.data
            in_id = id(inp)
            in_shape = inp.shape

            # Must be 4D tensor
            assert len(inp.shape) == 4

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

                in_shape = output_shape

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, in_shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])

                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, in_shape, inp.dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                elif parent_layer_names[0] == 'data':
                    # Input is constant.

                    # To NHWC
                    in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])

                    if in_data is not None:
                        in_data = np.transpose(inp.data, (0, 2, 3, 1))

                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input0' + format_prefix, inp.dtype,
                        in_shape, in_data)

            # output
            _output = func.outputs[0]
            output_name = layer_name + '_0'
            output_shape = (_output().shape[0], _output().shape[2],
                            _output().shape[3], _output().shape[1])
            output_id = tf_serializer.SerializeTensor(output_name,
                                                      _output().dtype,
                                                      output_shape, None)

            tf_serializer.RegisterTensorIdWithName(output_name, output_id)

            # block size
            block_size = func.r

            serialize_ops.SerializeOpSpaceToDepth(tf_serializer, input_id,
                                                  output_id, block_size)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, inp.dtype,
                [0, 3, 1, 2], reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'Cast':

            inp = func.inputs[0]

            # input
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input{}'.format(0), inp.dtype, inp.shape,
                    inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[i]))
                    raise

            # output
            _output = func.outputs[0]
            output_id = tf_serializer.SerializeTensor(layer_name,
                                                      _output().dtype,
                                                      _output().shape, None)

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpCast(tf_serializer, input_id, output_id)

        elif func.label == 'BatchNormalization':

            # TODO(LTE): Gamma

            # Decomposed into
            #
            # b * (gamma / sqrt(v))
            #

            if len(func.inputs) <= 3:
                # Guess F.batch_normalization
                x = func.inputs[0].get_variable().array
                mean = x.mean(axis=func.axis)
                var = x.var(axis=func.axis)
                print('mean', mean)
                print('varn', var)
            else:
                # guess F.fixed_batch_normalization
                x = func.inputs[3].get_variable().array
                mean = x.mean(axis=func.axis)
                var = x.var(axis=func.axis)
                print('mean', mean)
                print('varn', var)

            raise

            inp = func.inputs[0]

            # input
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input{}'.format(0), inp.dtype, inp.shape,
                    inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[i]))
                    raise

            # output
            outp = func.outputs[0]
            output_id = tf_serializer.SerializeTensor(layer_name,
                                                      outp().dtype,
                                                      outp().shape, None)

            tf_serializer.RegisterTensorIdWithVariableId(id(outp()), output_id)

            serialize_ops.SerializeOpBatchNorm(tf_serializer, input_id,
                                               output_id)

        elif func.label == 'LocalResponseNormalization':

            inp = func.inputs[0]
            in_id = id(inp)

            # Must be 4D tensor
            assert len(inp.shape) == 4

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])

                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, inp.shape, inp.dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                elif parent_layer_names[0] == 'data':
                    # Input is constant.

                    # To NHWC
                    in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])

                    in_data = inp.data
                    if in_data is not None:
                        in_data = np.transpose(inp.data, (0, 2, 3, 1))

                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input0' + format_prefix, inp.dtype,
                        in_shape, in_data)

            # output
            _output = func.outputs[0]
            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name,
                                                      _output().dtype,
                                                      _output().shape, None)

            # params
            radius = int(
                func.n) // 2  # Width(Chainer) to radius(TensorFlow) conversion
            bias = float(func.k)
            alpha = float(func.alpha)
            beta = float(func.beta)

            tf_serializer.RegisterTensorIdWithName(output_name, output_id)

            serialize_ops.SerializeOpLocalResponseNormalization(
                tf_serializer, input_id, output_id, radius, bias, alpha, beta)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id,
                _output().shape, inp.dtype, [0, 3, 1, 2], reshaped_shape,
                layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == 'NormalizeL2':

            inp = func.inputs[0]

            # tflite only supports the last dimension for axis
            if isinstance(func.axis, tuple):
                # Use the first one
                axis = func.axis[0]
            else:
                axis = func.axis
            if axis != (len(inp.shape) - 1):
                logger.fatal(
                    'axis must be the last dimension of input Tensor, but got axis = {}, dim = {}'
                    .format(axis, len(inp.shape)))
                raise

            # input
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input{}'.format(0), inp.dtype, inp.shape,
                    inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[i]))
                    raise

            # output
            outp = func.outputs[0]
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      outp().dtype,
                                                      outp().shape, None)

            tf_serializer.RegisterTensorIdWithVariableId(id(outp()), output_id)

            serialize_ops.SerializeOpL2Normalization(tf_serializer, input_id,
                                                     output_id)

        elif func.label == 'ELU':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpELU(tf_serializer, input_id, output_id)

        elif func.label == 'Sigmoid':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            # Use Logistic in tflite
            serialize_ops.SerializeOpLogistic(tf_serializer, input_id,
                                              output_id)

        elif func.label == 'ReLU':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]
            logger.info("ReLU.input.id = {}".format(id(inp)))
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("ReLU.output.id = {}".format(id(_output())))
            logger.info("ReLU.output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpReLU(tf_serializer, input_id, output_id)

        elif func.label == 'LeakyReLU':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            # alpha(slope)
            alpha = func.slope
            logger.info('leay relu slope = {}'.format(alpha))

            serialize_ops.SerializeOpLeakyReLU(tf_serializer, input_id,
                                               output_id, alpha)

        elif func.label == 'Softmax':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]

            assert len(inp.shape) >= 2

            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            # axis
            axis = func.axis

            # tflite uses las dim as an axis.
            assert axis == (len(inp.shape) - 1)

            # NOTE(LTE): Chainer does not have `beta` parameter in softmax.
            beta = 1.0

            serialize_ops.SerializeOpSoftmax(tf_serializer, input_id,
                                             output_id, beta)

        elif func.label == 'LogSoftmax':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            inp = func.inputs[0]

            assert len(inp.shape) >= 2

            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, 'float32', inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            # axis
            axis = func.axis

            # tflite uses las dim as an axis.
            assert axis == (len(inp.shape) - 1)

            serialize_ops.SerializeOpLogSoftmax(tf_serializer, input_id,
                                                output_id)

        elif func.label == 'Vstack':

            assert (len(func.inputs) >= 2)

            # TODO(LTE): Support non-float32 type

            # input
            input_ids = []
            for (i, inp) in enumerate(func.inputs):

                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, 'float32', inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input{}'.format(i), 'float32',
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            if len(func.inputs[0].shape) == 1:
                axis = 0
                serialize_ops.SerializeOpPack(tf_serializer, input_ids,
                                              output_id, axis)
            else:
                axis = 0
                serialize_ops.SerializeOpConcatenation(tf_serializer,
                                                       input_ids, output_id,
                                                       axis)

        elif func.label == 'Hstack':

            assert (len(func.inputs) >= 2)

            # TODO(LTE): Support non-float32 type

            # input
            input_ids = []
            for (i, inp) in enumerate(func.inputs):

                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input{}'.format(i), inp.dtype,
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            if len(func.inputs[0].shape) == 1:
                axis = 0
                serialize_ops.SerializeOpConcatenation(tf_serializer,
                                                       input_ids, output_id,
                                                       axis)
            else:
                axis = 1
                serialize_ops.SerializeOpConcatenation(tf_serializer,
                                                       input_ids, output_id,
                                                       axis)

        elif func.label == 'Pad':

            assert (len(func.inputs) == 1)
            assert func.mode == 'constant'

            print(func.pad_bw)

            pad_value = float(0.0)
            if 'constant_values' in func.keywords:
                values = func.keywords['constant_values']
                if not isinstance(values, int) and len(values) > 1:
                    raise ValueError(
                        'Currently scalar constant valueues is supported for Pad '
                        'operation')
                elif not isinstance(values, int):
                    pad_value = float(values[0])
                else:
                    pad_value = float(values)

            print('pad value = ', pad_value)

            new_shape = func.outputs[0]().shape
            print('pad new_shape = ', new_shape)

            inp = func.inputs[0]
            in_shape = inp.shape
            in_data = inp.data

            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, in_shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', in_shape, in_data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # create a constant value tensor for padding.

            # tflite = 2D tensor with [begin, end]x ndim. For example:
            # [[pad0_b, pad0_e],
            #  [pad1_b, pad1_e],
            #  [pad2_b, pad2_e],
            #  [pad3_b, pad3_e]]
            print('func_pad = ', func.pad_bw)
            padding_values = []
            for pad_bw in func.pad_bw:
                if isinstance(pad_bw, np.ndarray):
                    padding_values.append([pad_bw[0], pad_bw[1]])
                else:
                    padding_values.append([pad_bw, pad_bw])

            print(padding_values)

            # paddig tensor must have same array length for the first axis with input tensor
            padding = np.array(padding_values, np.int32)

            print('padding.shape = ', padding.shape)
            print('padding = ', padding)
            padding_id = tf_serializer.SerializeTensor(layer_name + '_padding',
                                                       inp.dtype,
                                                       padding.shape, padding)

            constant_id = -1
            if abs(float(pad_value)) > 1e-6:
                # create a constant value tensor used for padding value.
                constant_value = np.array([pad_value], np.float32)
                constant_id = tf_serializer.SerializeTensor(
                    layer_name + '_constant_value', constant_value.dtype,
                    constant_value.shape, constant_value)

            # output
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      inp.dtype, new_shape,
                                                      None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(func.outputs[0]), output_id)

            serialize_ops.SerializeOpPad(tf_serializer, input_id, output_id,
                                         padding_id, constant_id)

        elif func.label == 'Reshape':

            raise bora

            assert (len(func.inputs) == 1)

            new_shape = func.outputs[0]().shape

            logger.info('Reshape : {}'.format(new_shape))

            inp = func.inputs[0]
            if inp.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    inp.name, inp.dtype, inp.shape, None)
                self.inputs[inp.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', inp.dtype, inp.shape, inp.data)
            else:
                input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      inp.dtype, new_shape,
                                                      None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)
            serialize_ops.SerializeOpReshape(tf_serializer, input_id,
                                             output_id, new_shape)

        elif func.label == 'ResizeImages':

            # Assume float32 image
            # TODO(LTE): Support multiple channels

            for _input in func.inputs:
                logger.info('ResizeImages in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            print('resize_images', func.inputs)
            print('output_shape', func.out_H, func.out_W)

            inp = func.inputs[0]
            assert len(inp.shape) == 4

            in_shape = inp.shape
            in_data = inp.data
            in_id = id(inp)

            in_tensor_id = tf_serializer.FindTensorIdByVariableId(in_id)

            if in_tensor_id:
                # Input is variable

                logger.info('insert Reshape for NCHW to NHWC')
                # Convert from Chainer NCHW to tflite NHWC
                output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])
                input_id = self._insertOpTransposeTensor(
                    tf_serializer, in_tensor_id, inp.shape, inp.dtype, [0, 2, 3, 1],
                    output_shape, layer_name + '_to_nhwc', '_nhwc')

            else:
                # Input is placeholder or constant.

                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id

                    logger.info('insert Reshape for NCHW to NHWC')
                    # Convert from Chainer NCHW to tflite NHWC
                    _output_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                     inp.shape[1])

                    input_id = self._insertOpTransposeTensor(
                        tf_serializer, input_id, inp.shape, inp.dtype,
                        [0, 2, 3, 1], _output_shape, layer_name + '_to_nhwc',
                        '_nhwc')

                elif parent_layer_names[0] == 'data':
                    # Input is constant.

                    # To NHWC
                    in_shape = (inp.shape[0], inp.shape[2], inp.shape[3],
                                inp.shape[1])

                    in_data = inp.data
                    if in_data is not None:
                        in_data = np.transpose(inp.data, (0, 2, 3, 1))

                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input0' + format_prefix, inp.dtype,
                        in_shape, in_data)

            # new_shape(1D tensor with 2 elements)
            new_shape_name = layer_name + '_new_shape'
            new_shape_id = tf_serializer.SerializeTensor(
                new_shape_name, 'int32', [2],
                np.array([func.out_H, func.out_W], dtype=np.int32))

            # output
            _output = func.outputs[0]

            # NCHW -> NHWC
            output_shape = (_output().shape[0], _output().shape[2],
                            _output().shape[3], _output().shape[1])

            logger.info("output.shape = {}".format(output_shape))
            print('len(shape) = {}'.format(len(output_shape)))
            print('ty(shape) = {}'.format(type(output_shape)))
            print('resize_images.out_shape = {}'.format(output_shape))
            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name, inp.dtype,
                                                      output_shape, None)
            tf_serializer.RegisterTensorIdWithName(output_name, output_id)

            serialize_ops.SerializeOpResizeBilinear(tf_serializer, input_id,
                                                    output_id, new_shape_id)

            # tflite NHWC to Chainer NCHW
            reshaped_shape = _output().shape

            reshaped_output_id = self._insertOpTransposeTensor(
                tf_serializer, output_id, output_shape, inp.dtype,
                [0, 3, 1, 2], reshaped_shape, layer_name + '_to_nchw', '_nchw')

            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), reshaped_output_id)

        elif func.label == '_ + _':
            # Add

            if len(func.inputs) != 2:
                logger.fatal(
                    'The number of inputs for `Add` op must be two(2) but got got {}'
                    .format(len(func.inputs)))

            for _input in func.inputs:
                logger.info('Add in %s(id %d)', self._get_parent_name(_input),
                            id(_input))

            input_ids = []

            for (i, inp) in enumerate(func.inputs):
                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    # Constant
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input'.format(i), inp.data.dtype,
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("Add. output.shape = {}".format(_output().shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name,
                                                      func.inputs[0].dtype,
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpAdd(tf_serializer, input_ids[0],
                                         input_ids[1], output_id)

        elif func.label == '_ - _':
            # Sub

            if len(func.inputs) != 2:
                logger.fatal(
                    'The number of inputs for `Sub` op must be two(2) but got got {}'
                    .format(len(func.inputs)))

            for _input in func.inputs:
                logger.info('Sub in %s(id %d)', self._get_parent_name(_input),
                            id(_input))

            input_ids = []

            for (i, inp) in enumerate(func.inputs):
                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    # Constant
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input'.format(i), inp.data.dtype,
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name, 'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpSub(tf_serializer, input_ids[0],
                                         input_ids[1], output_id)

        elif func.label == '_ * _':
            # Mul

            if len(func.inputs) != 2:
                logger.fatal(
                    'The number of inputs for `Mul` op must be two(2) but got got {}'
                    .format(len(func.inputs)))

            for _input in func.inputs:
                logger.info('Mul in %s(id %d)', self._get_parent_name(_input),
                            id(_input))

            input_ids = []

            for (i, inp) in enumerate(func.inputs):
                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    # Constant
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input'.format(i), inp.data.dtype,
                        inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindTensorIdByVariableId(id(inp))
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name, 'float32',
                                                      _output().shape, None)
            tf_serializer.RegisterTensorIdWithVariableId(
                id(_output()), output_id)

            serialize_ops.SerializeOpMul(tf_serializer, input_ids[0],
                                         input_ids[1], output_id)

        elif func.label == 'Dropout':

            assert (len(func.inputs) > 0)

            # Require non-trivial conversion for Dropput
            convert_dropout.ConvertDropout(self, tf_serializer, func.inputs[0],
                                           layer_name, parent_layer_names[0],
                                           func.dropout_ratio)

        else:

            logger.fatal("Unknown or unsupported function/link : %s",
                         func.label)
            raise

    def __call__(self, _inputs, outputs):

        # register list of input names
        for _inp in _inputs:
            assert _inp.name is not None
            self.input_names.append(_inp.name)

        logger.info('input names = {}'.format(self.input_names))
        #logger.info('outputs = {}'.format(outputs))

        dumped_list = _dump_graph(outputs)
        # logger.debug('dumped_list = %s', dumped_list)
        assert (len(dumped_list) > 0)

        # Run DepthwiseConvolution2D foldering
        dumped_list = self._fold_depthwise_conv2d(dumped_list)

        # Run Transpose + Conv2d foldering
        dumped_list = self._fold_transpose_and_conv2d(dumped_list)

        logger.info("-- list of nodes -------------------")
        for i, l in enumerate(dumped_list):
            logger.info("=============================")
            logger.info("{}, {}, {}".format(i, id(l), l.label))
            for idx, inp in enumerate(l.inputs):
                logger.info("  input[{}].shape = {}".format(idx, inp.shape))
            for idx, outp in enumerate(l.outputs):
                # output is a weakref
                logger.info("  output[{}].shape = {}".format(idx, outp().shape))
        logger.info("------------------------------------")

        f = None
        tf_serializer = TensorFlowLiteSerializer()

        logger.info('outputs = %s', outputs[0].label)

        try:
            for i in dumped_list:
                self.dump_function_object(i, tf_serializer)

            # Flattern
            input_ids = [self.inputs[name] for name in self.inputs]

            # TODO(LTE): Find output in more rubust way.
            output_ids = [tf_serializer.num_tensor_ids - 1]

            subgraph = tf_serializer.SerializeSubGraph(input_ids, output_ids)
            tfmodel = tf_serializer.SerializeModel(subgraph)

            buf = tf_serializer.GetOutput(tfmodel)

            tflitemodel_filepath = self.tflitemodel

            if tflitemodel_filepath is None:
                tflitemodel_filepath = 'chainer_model.tflite'

            f = open(tflitemodel_filepath, 'wb')
            f.write(buf)
            f.close()

            logger.info("Wrote a file: {} ({} bytes)".format(
                tflitemodel_filepath, len(buf)))

        finally:
            if f is not None:
                f.close()


def export(model, args, filename):

    # forward eval

    #
    # Transform input(args) to chainer.Variable type and assign unique name.
    #
    inputs = []
    if isinstance(args, tuple):
        args = list(args)

    if isinstance(args, list):
        for i, arg in enumerate(args):
            if isinstance(arg, chainer.get_array_types()):
                input_name = 'input{}'.format(i)
                args[i] = chainer.Variable(arg, name=input_name)
            else:
                assert isinstance(arg, chainer.Variable)
                if args[i].name is None:
                    # assign name
                    args[i].name = 'input{}'.format(i)

            inputs.append(args[i])

        outputs = model(*args)
    elif isinstance(args, chainer.get_array_types()):
        args = chainer.Variable(args, name='input0')
        inputs.append(args)
        outputs = model(args)

    elif isinstance(args, chainer.Variable):
        if args.name is None:
            # assign name
            args.name = 'input0'
        inputs.append(args)
        outputs = model(args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list, tuple, '
            'numpy array, or Chainer Variable. But a {} object was '
            'given.'.format(type(args)))

    assert len(inputs) > 0

    for inp in inputs:
        assert inp.name is not None
        logger.info('DBG: Chainer input name = {}'.format(inp.name))

    if isinstance(outputs, variable.Variable):
        outputs = [outputs]
    assert isinstance(outputs, (tuple, list))

    logger.info('# of inputs = {}'.format(len(inputs)))
    logger.info('# of outputs = {}'.format(len(outputs)))

    # Assign name
    for i, outp in enumerate(outputs):
        assert isinstance(outp, variable.Variable)
        if outp.name is None:
            outp.name = 'output{}'.format(i)

        logger.info('output name[{}] = {}'.format(i, outp.name))

    converter = TensorFlowLiteConverter(filename)
    converter(inputs, outputs)

    # Chainer's result
    return inputs, outputs
