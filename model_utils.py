import numpy as np
import tensorflow as tf
from math import ceil, floor, log

def count_layers(height, width):
    size = min(height, width)
    cnt = 0
    while size > 4:
        size = ceil(size / 2)
        cnt += 1
    return cnt, int(size)

def valid_kernel_and_strides(kernel_size, strides):
    # Validation of kernel_size and strides
    if type(kernel_size) == tuple or type(kernel_size) == list:
        assert len(kernel_size) == 2, "KernelSizeLengthError"
        kernel_height = kernel_size[0]
        kernel_width = kernel_size[1]
    elif type(kernel_size) == int:
        kernel_height = kernel_width = kernel_size
    else:
        assert False, "KernelSizeTypeError"
    if type(strides) == tuple or type(strides) == list:
        assert len(strides) == 2 or len(strides) == 4, "StridesLengthError"
        if len(strides) == 2: _strides = [1, strides[0], strides[1], 1]
        else: _strides = strides
    elif type(strides) == int:
        _strides = [1, strides, strides, 1]
    else:
        assert False, "StridesTypeError"

    return kernel_height, kernel_width, _strides

class batch_norm(object):
    def __init__(self, momentum=0.9, epsilon=1e-5, name="batch_norm"):
        with tf.variable_scope(name):
            self.momentum = momentum
            self.epsilon = epsilon
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
                x,
                decay=self.momentum,
                updates_collections=None,
                epsilon=self.epsilon,
                scale=True,
                is_training=train,
                scope=self.name
                )

def instance_norm(input_tensor, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input_tensor.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_tensor, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_tensor - mean) * inv
        return scale * normalized + offset

def conv2d(
        input_tensor, output_dim,
        kernel_size=(5, 5),
        strides=(2, 2),
        stddev=0.02,
        name="conv2d",
        padding="SAME",
        return_weight=False
        ):
    with tf.variable_scope(name):
        kernel_height, kernel_width, _strides = valid_kernel_and_strides(kernel_size, strides)
        input_channel = input_tensor.get_shape()[-1]
        
        W = tf.get_variable('W', [kernel_height, kernel_width, input_channel, output_dim], 
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_tensor, W, strides=_strides, padding=padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape()) 

        if return_weight: return conv, [W, b]
        else: return conv

def deconv2d(
        input_tensor, output_shape,
        kernel_size=(5, 5),
        strides=(2, 2),
        stddev=0.02,
        name="deconv2d",
        return_weight=False
        ):
    with tf.variable_scope(name):
        kernel_height, kernel_width, _strides = valid_kernel_and_strides(kernel_size, strides)
        output_channel = output_shape[-1]
        input_channel = input_tensor.get_shape()[-1]

        W = tf.get_variable('W', [kernel_height, kernel_width, output_channel, input_channel],
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_channel], initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.conv2d_transpose(input_tensor, W, output_shape=output_shape, strides=_strides)
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        if return_weight: return deconv, [W, b]
        else: return deconv

def residule_block(input_tensor, dim, kernel_size=(3, 3), strides=(1, 1), name="residule_block"):
    padding = [int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2)]
    y = tf.pad(input_tensor, [[0, 0], padding, padding, [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, kernel_size, strides, padding='VALID', name=name+'_conv1'), name+"_in1")
    y = tf.pad(tf.nn.relu(y), [[0, 0], padding, padding, [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, kernel_size, strides, padding='VALID', name=name+'_conv2'), name+"_in2")
    return y + input_tensor

def linear(
        input_tensor, output_dim,
        stddev=0.02,
        bias_initial_value=0.0,
        name="linear",
        return_weight=False
        ):
    with tf.variable_scope(name):
        assert len(input_tensor.get_shape()) <= 2, "NotFlattenError"
        assert len(input_tensor.get_shape()) > 1, "DimensionError"
        input_dim = input_tensor.get_shape()[-1]

        W = tf.get_variable('W', [input_dim, output_dim],
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias_initial_value))

        output = tf.matmul(input_tensor, W) + b

        if return_weight: return output, [W, b]
        else: return output

def LeakyReLU(x, leak=0.2):
    return tf.maximum(x, leak*x)
