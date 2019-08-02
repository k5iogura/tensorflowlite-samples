# -*- coding: utf-8 -*-

from . import tflite

from .tflite import FullyConnectedOptions
from .tflite import Operator
from .tflite import OperatorCode
from .tflite import BuiltinOperator
from .tflite import BuiltinOptions
from .tflite import AddOptions
from .tflite import SubOptions
from .tflite import MulOptions
from .tflite import DivOptions
from .tflite import DivOptions
from .tflite import FloorDivOptions
from .tflite import ReshapeOptions
from .tflite import ResizeBilinearOptions
from .tflite import ResizeNearestNeighborOptions

from .tflite import Conv2DOptions
from .tflite import Pool2DOptions
from .tflite import DepthwiseConv2DOptions

from .tflite import PadOptions
from .tflite import LeakyReluOptions
from .tflite import SoftmaxOptions
from .tflite import PackOptions
from .tflite import ConcatenationOptions
from .tflite import SqueezeOptions

from .tflite import L2NormOptions
from .tflite import LocalResponseNormalizationOptions

from .tflite import SpaceToDepthOptions

from .tflite import TransposeOptions
from .tflite import TileOptions
from .tflite import ExpandDimsOptions

from .tflite import ActivationFunctionType
from .tflite import Padding


def SerializeOpFullyConnected(serializer, fused_activation_function, input_id,
                              output_id, W_id, b_id):

    serializer.logger.info(
        "fully_connected. input = {}, output = {}, W = {}, b = {}".format(
            input_id, output_id, W_id, b_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    tflite.FullyConnectedOptions.FullyConnectedOptionsStart(serializer.builder)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tf_options = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 3
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(b_id)
    serializer.builder.PrependInt32(W_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeConv2D(serializer, input_id, filter_id, bias_id, output_id,
                    fused_activation_function, padding, stride, dilations):
    """Serialize conv2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        filter_id(int): Filter Tensor id
        bias_id(int): Bias Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        dilations([int]): [dilation_w_factor, dilation_h_factor].

    """

    serializer.logger.info(
        "conv2d. input = {}, filter = {}, bias = {}, output = {}, fused_activation_function = {}, stride = {}, dilations = {}"
        .format(input_id, filter_id, bias_id, output_id,
                fused_activation_function, stride, dilations))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.CONV_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise

    tflite.Conv2DOptions.Conv2DOptionsStart(serializer.builder)
    tflite.Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.Conv2DOptions.Conv2DOptionsAddPadding(serializer.builder,
                                                 padding_type)
    tflite.Conv2DOptions.Conv2DOptionsAddStrideW(serializer.builder, stride[0])
    tflite.Conv2DOptions.Conv2DOptionsAddStrideH(serializer.builder, stride[1])
    tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(
        serializer.builder, dilations[0])
    tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(
        serializer.builder, dilations[1])
    tf_options = tflite.Conv2DOptions.Conv2DOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 3
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(bias_id)
    serializer.builder.PrependInt32(filter_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.Conv2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeTransposeConv(serializer, output_shape_id, weights_id, input_id,
                           output_id, padding, stride):
    """Serialize TransposeConv.

    Args:
        serializer: tflite serializer.
        output_shape_id(int): Id for a Tensor contains output shape size.
        weights_id(int): Weights Tensor id.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].

    """

    serializer.logger.info(
        "transpose_conv. output_shape = {}, weights = {}, input = {}, output = {},  stride = {}, padding = {}"
        .format(output_shape_id, weights_id, input_id, output_id, stride,
                padding))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.TRANSPOSE_CONV)

    # Options
    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise

    tflite.TransposeConvOptions.TransposeConvOptionsStart(serializer.builder)
    tflite.TransposeConvOptions.TransposeConvOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.TransposeConvOptions.TransposeConvOptionsAddPadding(
        serializer.builder, padding_type)
    tflite.TransposeConvOptions.TransposeConvOptionsAddStrideW(
        serializer.builder, stride[0])
    tflite.TransposeConvOptions.TransposeConvOptionsAddStrideH(
        serializer.builder, stride[1])
    tf_options = tflite.TransposeConvOptions.Conv2DOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 3
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    serializer.builder.PrependInt32(weights_id)
    serializer.builder.PrependInt32(output_shape_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.TransposeConvOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeDepthwiseConv2D(serializer, input_id, filter_id, bias_id,
                             output_id, fused_activation_function, padding,
                             stride, dilations, multiplier):
    """Serialize depthwise_conv2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        filter_id(int): Filter Tensor id
        bias_id(int): Bias Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        dilations([int]): [dilation_w_factor, dilation_h_factor].
        multiplier(int): Multiplier.

    """

    serializer.logger.info(
        "depthwise_conv2d. input = {}, filter = {}, bias = {}, output = {}, fused_activation_function = {}, stride = {}, multiplier = {}"
        .format(input_id, filter_id, bias_id, output_id,
                fused_activation_function, stride, multiplier))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise

    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(
        serializer.builder)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(
        serializer.builder, padding_type)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(
        serializer.builder, stride[0])
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(
        serializer.builder, stride[1])
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(
        serializer.builder, dilations[0])
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(
        serializer.builder, dilations[1])
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(
        serializer.builder, multiplier)
    tf_options = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 3
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(bias_id)
    serializer.builder.PrependInt32(filter_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeAveragePooling2D(serializer, input_id, output_id,
                              fused_activation_function, padding, stride,
                              filter_size):
    """Serialize average_pooling_2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        filter_size([int]): [filter_width, filter_height].

    """

    serializer.logger.info(
        "average_pooling_2d. input = {}, output = {}, fused_activation_function = {}, stride = {}, filter_size = {}"
        .format(input_id, output_id, fused_activation_function, stride,
                filter_size))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.AVERAGE_POOL_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise

    tflite.Pool2DOptions.Pool2DOptionsStart(serializer.builder)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(serializer.builder,
                                                 padding_type)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(serializer.builder, stride[0])
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(serializer.builder, stride[1])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(serializer.builder,
                                                     filter_size[0])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(serializer.builder,
                                                      filter_size[1])
    tf_options = tflite.Pool2DOptions.Pool2DOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeMaxPooling2D(serializer, input_id, output_id,
                          fused_activation_function, padding, stride,
                          filter_size):
    """Serialize max_pooling_2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        filter_size([int]): [filter_width, filter_height].

    """

    serializer.logger.info(
        "max_pooling_2d. input = {}, output = {}, fused_activation_function = {}, stride = {}, filter_size = {}"
        .format(input_id, output_id, fused_activation_function, stride,
                filter_size))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.MAX_POOL_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise

    tflite.Pool2DOptions.Pool2DOptionsStart(serializer.builder)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(serializer.builder,
                                                 padding_type)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(serializer.builder, stride[0])
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(serializer.builder, stride[1])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(serializer.builder,
                                                     filter_size[0])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(serializer.builder,
                                                      filter_size[1])
    tf_options = tflite.Pool2DOptions.Pool2DOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpResizeBilinear(serializer, input_id, output_id, new_shape_id):

    # NOTE(LTE): Chainer supports bilinear interpolation only.
    # Map to resize_bilinear + align_corners = true.
    # For more details about resize_images,
    # See https://github.com/chainer/onnx-chainer/issues/147

    serializer.logger.info(
        "resize_images. input = {}, output = {}, new_shape = {}".format(
            input_id, output_id, new_shape_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESIZE_BILINEAR)

    # Options
    # (`align_corners` == true) matches the Chainer's result.
    tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(serializer.builder)
    tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(
        serializer.builder, True)
    tf_options = tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(
        serializer.builder)

    # Inputs
    # new_shape first.
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(new_shape_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ResizeBilinearOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpResizeNearestNeighbor(serializer, input_id, output_id,
                                     new_shape_id):

    serializer.logger.info(
        "resize_nearest_neighbor. input = {}, output = {}, new_shape = {}".
        format(input_id, output_id, new_shape_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR)

    # Options
    # TODO(LTE): Do we need to set align_corners?
    tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(serializer.builder)
    #tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(
    #    serializer.builder, True)
    tf_options = tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(
        serializer.builder)

    # Inputs
    # new_shape first.
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(new_shape_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ResizeNearestNeighborOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpAdd(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.ADD)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    activation_function = 0  # 'NONE'
    tflite.AddOptions.AddOptionsStart(serializer.builder)
    tflite.AddOptions.AddOptionsAddFusedActivationFunction(
        serializer.builder, activation_function)
    tf_options = tflite.AddOptions.AddOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.AddOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSub(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SUB)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    activation_function = 0  # 'NONE'
    tflite.SubOptions.SubOptionsStart(serializer.builder)
    tflite.SubOptions.SubOptionsAddFusedActivationFunction(
        serializer.builder, activation_function)
    tf_options = tflite.SubOptions.SubOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.SubOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpMul(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.MUL)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    activation_function = 0  # 'NONE'
    tflite.MulOptions.MulOptionsStart(serializer.builder)
    tflite.MulOptions.MulOptionsAddFusedActivationFunction(
        serializer.builder, activation_function)
    tf_options = tflite.MulOptions.MulOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.MulOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpDiv(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.DIV)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    activation_function = 0  # 'NONE'
    tflite.DivOptions.DivOptionsStart(serializer.builder)
    tflite.DivOptions.DivOptionsAddFusedActivationFunction(
        serializer.builder, activation_function)
    tf_options = tflite.DivOptions.DivOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.DivOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpRsqrt(serializer, input_id, output_id):
    """Serialize Rsqrt op.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RSQRT)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpFloor(serializer, input_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.FLOOR)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpFloorDiv(serializer, x_id, y_id, output_id):

    # floor(x / y)

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.FLOOR_DIV)

    # Options
    tflite.FloorDivOptions.FloorDivOptionsStart(serializer.builder)
    # No parameter
    tf_options = tflite.FloorDivOptions.FloorDivOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.FloorDivOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpCeil(serializer, input_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.CEIL)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpPad(serializer, input_id, output_id, padding_id, constant_id):
    """Serialize Pad.

    Args:

        input_id (int): Input tensor id.
        output_id (int): Output tensor id.
        padding_id (int): Tensor id which contains padding size(2x2 shape).
        constant_id (int): Optional constant value for padded area.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.PADV2)

    # Options
    # Currently PadOptions has empty parameter.
    tflite.PadOptions.PadOptionsStart(serializer.builder)
    tf_options = tflite.PadOptions.PadOptionsEnd(serializer.builder)

    # Inputs
    if constant_id == -1:
        num_inputs = 2
        tflite.Operator.OperatorStartInputsVector(serializer.builder,
                                                  num_inputs)
        serializer.builder.PrependInt32(padding_id)
        serializer.builder.PrependInt32(input_id)
        tf_inputs = serializer.builder.EndVector(num_inputs)
    else:
        # Even though constant value tensor is not described in tflite document,
        # tflite interpreter implementation supoorts it.
        num_inputs = 3
        tflite.Operator.OperatorStartInputsVector(serializer.builder,
                                                  num_inputs)
        serializer.builder.PrependInt32(constant_id)
        serializer.builder.PrependInt32(padding_id)
        serializer.builder.PrependInt32(input_id)
        tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.PadOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpReshape(serializer, input_id, output_id, new_shape):
    """Serialize Reshape function.

    Args:

        new_shape ([int]): New shape.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESHAPE)

    # Options
    tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
        serializer.builder, len(new_shape))
    for i in reversed(new_shape):
        serializer.builder.PrependInt32(i)
    tf_new_shape = serializer.builder.EndVector(len(new_shape))

    tflite.ReshapeOptions.ReshapeOptionsStart(serializer.builder)
    tflite.ReshapeOptions.ReshapeOptionsAddNewShape(serializer.builder,
                                                    tf_new_shape)
    tf_options = tflite.ReshapeOptions.ReshapeOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpELU(serializer, input_id, output_id):
    """Serialize ELU op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id

    Returns:
        tflite.Operator

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.ELU)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpLogistic(serializer, input_id, output_id):
    """Serialize Logistic op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id

    Returns:
        tflite.Operator

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.LOGISTIC)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpReLU(serializer, input_id, output_id):
    """Serialize ReLU op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id

    Returns:
        tflite.Operator

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RELU)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpLeakyReLU(serializer, input_id, output_id, alpha):
    """Serialize LeakyReLU op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id
        alpha(float): Slope of the activation at x < 0 (provided alpha <= 1)

    Returns:
        tflite.Operator

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.LEAKY_RELU)

    # Options
    tflite.LeakyReluOptions.LeakyReluOptionsStart(serializer.builder)
    tflite.LeakyReluOptions.LeakyReluOptionsAddAlpha(serializer.builder, alpha)
    tf_options = tflite.LeakyReluOptions.LeakyReluOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.LeakyReluOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSoftmax(serializer, input_id, output_id, beta):
    """Serialize Softmax op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id
        beta(float): Scaling factor

    Returns:
        tflite.Operator

    """

    # Options
    tflite.SoftmaxOptions.SoftmaxOptionsStart(serializer.builder)
    tflite.SoftmaxOptions.SoftmaxOptionsAddBeta(serializer.builder, beta)
    tf_options = tflite.SoftmaxOptions.SoftmaxOptionsEnd(serializer.builder)

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SOFTMAX)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.SoftmaxOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpLogSoftmax(serializer, input_id, output_id):
    """Serialize LogSoftmax op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id

    Returns:
        tflite.Operator

    """

    # TODO(LTE): Support parameters for log_softmax op

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.LOG_SOFTMAX)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpReshape(serializer, input_id, output_id, new_shape):
    """Serialize Reshape function.

    Args:

        input_id (int): Input Tensor id.
        output_id (int): Output Tensor id.
        new_shape ([int]): New shape.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESHAPE)

    # Options
    tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
        serializer.builder, len(new_shape))
    for i in reversed(new_shape):
        serializer.builder.PrependInt32(i)
    tf_new_shape = serializer.builder.EndVector(len(new_shape))

    tflite.ReshapeOptions.ReshapeOptionsStart(serializer.builder)
    tflite.ReshapeOptions.ReshapeOptionsAddNewShape(serializer.builder,
                                                    tf_new_shape)
    tf_options = tflite.ReshapeOptions.ReshapeOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpPack(serializer, input_ids, output_id, axis):
    """Serialize Pack function.

    Args:

        input_id ([int]): List of input Tensor id.
        output_id (int): Output Tensor id.
        axis (int): Axis for packing.

    """

    serializer.logger.info("pack. inputs = {}, axis = {}, output = {}".format(
        input_ids, axis, output_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.PACK)

    # `value_count` parameter should be same with len(input_ids)
    value_count = len(input_ids)
    assert value_count > 1

    # Options
    tflite.PackOptions.PackOptionsStart(serializer.builder)
    tflite.PackOptions.PackOptionsAddValuesCount(serializer.builder,
                                                 value_count)
    tflite.PackOptions.PackOptionsAddAxis(serializer.builder, axis)
    tf_options = tflite.PackOptions.PackOptionsEnd(serializer.builder)

    # Inputs
    # NOTE(LTE): 2nd input is an integer, not tensor id.
    num_inputs = value_count
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    for t_id in reversed(input_ids):
        serializer.builder.PrependInt32(t_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.PackOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpConcatenation(serializer, input_ids, output_id, axis):
    """Serialize Concatenation function.

    Args:

        input_id ([int]): List of input Tensor id.
        output_id (int): Output Tensor id.
        axis (int): Axis for packing.

    """

    serializer.logger.info(
        "concatenation. inputs = {}, axis = {}, output = {}".format(
            input_ids, axis, output_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.CONCATENATION)

    # `value_count` parameter should be same with len(input_ids)
    value_count = len(input_ids)
    assert value_count > 1

    # Options
    tflite.ConcatenationOptions.ConcatenationOptionsStart(serializer.builder)

    # TODO(LTE): Support FAF
    tflite.ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(
        serializer.builder,
        tflite.ActivationFunctionType.ActivationFunctionType.NONE)
    tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(
        serializer.builder, axis)
    tf_options = tflite.ConcatenationOptions.ConcatenationOptionsEnd(
        serializer.builder)

    # Inputs
    # NOTE(LTE): 2nd input is an integer, not tensor id.
    num_inputs = value_count
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    for t_id in reversed(input_ids):
        serializer.builder.PrependInt32(t_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ConcatenationOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpFill(serializer, input_id, constant_id, output_id):
    """Serialize Fill op.

    Args:

        input_id (int): Input tensor id.
        constant_id (int): A 0D tensor which contains constant value used for filling.
        output_id (int): Output Tensor id.

    """

    serializer.logger.info(
        "fill. input = {}, constant = {}, output = {}".format(
            input_id, constant_id, output_id))
    tf_opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.FILL)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(constant_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, tf_opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpTranspose(serializer, input_id, perm_id, output_id):
    """Serialize Transpose op.

    Args:

        input_id (int): Input Tensor id.
        perm_id (int): A tensor containing permutations(e.g. 1D tensor [0, 2, 3, 1])
        output_id (int): Output Tensor id.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.TRANSPOSE)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(perm_id)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.TransposeOptions.TransposeOptionsStart(serializer.builder)
    tf_options = tflite.TransposeOptions.TransposeOptionsEnd(
        serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.TransposeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpExpandDims(serializer, input_id, axis_id, output_id):
    """Serialize ExpandDim op.

    Args:

        input_id (int): Input tensor id.
        axis_id (int): A 0D tensor which contains `axis` value.
        output_id (int): Output Tensor id.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.EXPAND_DIMS)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(axis_id)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.ExpandDimsOptions.ExpandDimsOptionsStart(serializer.builder)
    tf_options = tflite.ExpandDimsOptions.ExpandDimsOptionsEnd(
        serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ExpandDimsOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpTile(serializer, input_id, multiples_id, output_id):
    """Serialize Tile op.

    Args:

        input_id (int): Input Tensor id.
        multiples_id (int): A Tensor containnig multiples.
        output_id (int): Output Tensor id.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.TILE)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(multiples_id)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.TileOptions.TileOptionsStart(serializer.builder)
    tf_options = tflite.TileOptions.TileOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.TileOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSqueeze(serializer, input_id, output_id, squeeze_dims):
    """Serialize Squeeze op.

    Args:

        input_id (int): Input tensor id.
        output_id (int): Output Tensor id.
        squeeze_dims([int]): List of squeeze dims

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SQUEEZE)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.SqueezeOptions.SqueezeOptionsStartSqueezeDimsVector(
        serializer.builder, len(squeeze_dims))
    for i in reversed(squeeze_dims):
        serializer.builder.PrependInt32(i)
    tf_dims = serializer.builder.EndVector(len(squeeze_dims))

    tflite.SqueezeOptions.SqueezeOptionsStart(serializer.builder)
    tflite.SqueezeOptions.SqueezeOptionsAddSqueezeDims(serializer.builder,
                                                       tf_dims)
    tf_options = tflite.SqueezeOptions.SqueezeOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.SqueezeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSplit(serializer, axis_id, input_id, output_ids, num_splits):
    """Serialize Split op.

    Args:

        axis_id (int): 0D tensor with axis information.
        input_id (int): Input tensor id.
        output_ids ([int]): List of output Tensor ids(subtenors built from the input tensors).
        num_splits(int): Number of splits.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SPLIT)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    # Axis first
    serializer.builder.PrependInt32(input_id)
    serializer.builder.PrependInt32(axis_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = len(output_ids)
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    for i in reversed(output_ids):
        serializer.builder.PrependInt32(i)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.SqueezeOptions.SqueezeOptionsStartSqueezeDimsVector(
        serializer.builder, len(squeeze_dims))
    for i in reversed(squeeze_dims):
        serializer.builder.PrependInt32(i)
    tf_dims = serializer.builder.EndVector(len(squeeze_dims))

    tflite.SplitOptions.SplitOptionsStart(serializer.builder)
    tflite.SplitOptions.SplitOptionsAddNumSplits(serializer.builder,
                                                 num_splits)
    tf_options = tflite.SplitOptions.SplitOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.SplitOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSpaceToDepth(serializer, input_id, output_id, block_size):
    """Serialize SpaceToDepth op.

    Args:

        input_id (int): Input tensor id.
        output_ids ([int]): List of output Tensor ids(subtenors built from the input tensors).
        block_size(int): Block size.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SPACE_TO_DEPTH)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    tflite.SpaceToDepthOptions.SpaceToDepthOptionsStart(serializer.builder)
    tflite.SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(
        serializer.builder, block_size)
    tf_options = tflite.SpaceToDepthOptions.SpaceToDepthOptionsEnd(
        serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.SpaceToDepthOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpCast(serializer, input_id, output_id):
    """Serialize Cast op.

    Args:

        input_id (int): Input tensor id.
        output_ids ([int]): List of output Tensor ids(subtenors built from the input tensors).

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.CAST)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # We won't need data type parameter for input and output.
    # So do not write out CastOptions

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpLocalResponseNormalization(serializer, input_id, output_id,
                                          radius, bias, alpha, beta):
    """Serialize LocalResponseNormalization op.

    Args:

        input_id (int): Input tensor id.
        output_id (int): Output tensor id.
        radius(int): Radius
        bias(float): Bias
        alpha(float): Alpha
        beta(float): Beta

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION)

    # Options
    tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsStart(
        serializer.builder)
    tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddRadius(
        serializer.builder, radius)
    tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddBias(
        serializer.builder, bias)
    tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddAlpha(
        serializer.builder, alpha)
    tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddBeta(
        serializer.builder, beta)
    tf_options = tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # We won't need data type parameter for input and output.
    # So do not write out CastOptions

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.LocalResponseNormalizationOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpL2Normalization(serializer, input_id, output_id):
    """Serialize L2Normalization op.

    Args:

        input_id (int): Input tensor id.
        output_id (int): Output tensor id.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.L2_NORMALIZATION)

    tflite.L2NormOptions.L2NormOptionsStart(serializer.builder)
    # Current tflite implentation does not support activation function.
    tflite.L2NormOptions.L2NormOptionsAddFusedActivationFunction(
        serializer.builder,
        tflite.ActivationFunctionType.ActivationFunctionType.NONE)
    tf_options = tflite.L2NormOptions.L2NormOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # We won't need data type parameter for input and output.
    # So do not write out CastOptions

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.L2NormOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op
