#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from packaging import version
import tensorflow as tf

# Check TensorFlow version
print("Detected TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, 'This code requires TensorFlow 2.0 or above.'

# Set eager execution
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


def identity_block(input_tensor, kernel_size, filters, stage, block, activation=True, include_batchnorm=False, short_cut=True):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    batchnorm_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1),
                               dilation_rate=(1, 1),
                               kernel_initializer='glorot_uniform',
                               name=conv_name_base + '2a')(input_tensor)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size,
                               dilation_rate=(1, 1),
                               padding='same',
                               kernel_initializer='glorot_uniform',
                               name=conv_name_base + '2b')(x)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1),
                               dilation_rate=(1, 1),
                               kernel_initializer='glorot_uniform',
                               name=conv_name_base + '2c')(x)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2c')(x)

    if short_cut:
        x = tf.keras.layers.add([x, input_tensor])

    if activation:
        x = tf.keras.layers.ReLU()(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), activation=True, include_batchnorm=False, short_cut=True):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    batchnorm_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides,
                               dilation_rate=(1, 1),
                               kernel_initializer='glorot_uniform',
                               name=conv_name_base + '2a')(input_tensor)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',
                               dilation_rate=(1, 1),
                               kernel_initializer='glorot_uniform',
                               name=conv_name_base + '2b')(x)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1),
                               kernel_initializer='glorot_uniform',
                               dilation_rate=(1, 1),
                               name=conv_name_base + '2c')(x)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '2c')(x)

    if short_cut:
        shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                                          dilation_rate=(1, 1),
                                          kernel_initializer='glorot_uniform',
                                          name=conv_name_base + '1')(input_tensor)
        if include_batchnorm:
            shortcut = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=bn_name_base + '1')(shortcut)

        x = tf.keras.layers.add([x, shortcut])

    if activation:
        x = tf.keras.layers.ReLU()(x)
    return x


def ResNet43_8s(input_shape, output_dim, include_batchnorm=False, batchnorm_axis=-1, prefix='', cutoff_early=False):
    """produces an hourglass FCN network, adapted to CoRL submission size.

    Regarding shapes, look at: https://www.tensorflow.org/api_docs/python/tf/keras/Input [excludes batch size]
    Here are the shape patterns, where I print shapes for the input, and after each conv_block or Conv2D call.

    Attention:
    (None, 320, 320, 6) (input shape)
    (None, 320, 320, 64)
    (None, 320, 320, 64)
    (None, 160, 160, 128)
    (None, 80, 80, 256)
    (None, 40, 40, 512)
    (None, 40, 40, 256)
    (None, 80, 80, 128)
    (None, 160, 160, 64)
    (None, 320, 320, 1)

    Transport, key module
    (None, 384, 224, 6) (input shape)
    (None, 384, 224, 64)
    (None, 384, 224, 64)
    (None, 192, 112, 128)
    (None, 96, 56, 256)
    (None, 48, 28, 512)
    (None, 48, 28, 256)
    (None, 96, 56, 128)
    (None, 192, 112, 64)
    (None, 384, 224, 3)

    Transport, query module, assumes cropping beforehand.
    (None, 64, 64, 6) (input shape)
    (None, 64, 64, 64)
    (None, 64, 64, 64)
    (None, 32, 32, 128)
    (None, 16, 16, 256)
    (None, 8, 8, 512)
    (None, 8, 8, 256)
    (None, 16, 16, 128)
    (None, 32, 32, 64)
    (None, 64, 64, 3)

    Here I ignore output after identity blocks, which produce tensors of the same size.

    Parameters
    ----------
    :input_shape: a tuple that specifies the shape for tf.keras.layers.Input. By default,
        it's (None,320,320,6) for Attention, (None,384,224,6) for Transport. The input is
        a tuple, not a tensor.
    :output_dim: a single scalar, which produces the number of channels of the output.
        For example, if it's set to 3, then if we pass kernels of size (None,64,64,6),
        the default, we get an output tensor of size (None,64,64,3).
    """
    input_data = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', name=prefix + 'conv1')(input_data)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=prefix + 'bn_conv1')(x)
    x = tf.keras.layers.ReLU()(x)

    if cutoff_early:
        x = conv_block(x, 5, [64, 64, output_dim], stage=2, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
        x = identity_block(x, 5, [64, 64, output_dim], stage=2, block=prefix + 'b', include_batchnorm=include_batchnorm)
        return input_data, x

    x = conv_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = conv_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = conv_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = conv_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = conv_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(x)

    x = conv_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(x)

    x = conv_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'b', include_batchnorm=include_batchnorm)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(x)

    x = conv_block(x, 3, [16, 16, output_dim], stage=9, block=prefix + 'a', strides=(1, 1), activation=False)
    output = identity_block(x, 3, [16, 16, output_dim], stage=9, block=prefix + 'b', activation=False)

    return input_data, output


def UNet43_8s(input_shape, output_dim, include_batchnorm=False, batchnorm_axis=-1, prefix='', cutoff_early=False):
    """produces an hourglass FCN network, adapted to CoRL submission size.

    Regarding shapes, look at: https://www.tensorflow.org/api_docs/python/tf/keras/Input [excludes batch size]
    Here are the shape patterns, where I print shapes for the input, and after each conv_block or Conv2D call.

    Attention:
    (None, 320, 320, 6) (input shape)
    (None, 320, 320, 64)
    (None, 320, 320, 64)
    (None, 160, 160, 128)
    (None, 80, 80, 256)
    (None, 40, 40, 512)
    (None, 40, 40, 256)
    (None, 80, 80, 128)
    (None, 160, 160, 64)
    (None, 320, 320, 1)

    Transport, key module
    (None, 384, 224, 6) (input shape)
    (None, 384, 224, 64)
    (None, 384, 224, 64)
    (None, 192, 112, 128)
    (None, 96, 56, 256)
    (None, 48, 28, 512)
    (None, 48, 28, 256)
    (None, 96, 56, 128)
    (None, 192, 112, 64)
    (None, 384, 224, 3)

    Transport, query module, assumes cropping beforehand.
    (None, 64, 64, 6) (input shape)
    (None, 64, 64, 64)
    (None, 64, 64, 64)
    (None, 32, 32, 128)
    (None, 16, 16, 256)
    (None, 8, 8, 512)
    (None, 8, 8, 256)
    (None, 16, 16, 128)
    (None, 32, 32, 64)
    (None, 64, 64, 3)

    Here I ignore output after identity blocks, which produce tensors of the same size.

    Parameters
    ----------
    :input_shape: a tuple that specifies the shape for tf.keras.layers.Input. By default,
        it's (None,320,320,6) for Attention, (None,384,224,6) for Transport. The input is
        a tuple, not a tensor.
    :output_dim: a single scalar, which produces the number of channels of the output.
        For example, if it's set to 3, then if we pass kernels of size (None,64,64,6),
        the default, we get an output tensor of size (None,64,64,3).
    """
    input_data = tf.keras.layers.Input(shape=input_shape)
    print(input_data.shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', name=prefix + 'conv1')(input_data)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=prefix + 'bn_conv1')(x)
    x = tf.keras.layers.ReLU()(x)
    print(x.shape)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x_320_160_64 = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = conv_block(x_320_160_64, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x_160_80_128 = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = conv_block(x_160_80_128, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x_80_40_256 = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x_80_40_256.shape)

    x = conv_block(x_80_40_256, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = conv_block(x, 3, [512, 512, 512], stage=17, block=prefix + 'a', strides=(2, 2), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [512, 512, 512], stage=17, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    global_feat = tf.nn.avg_pool(x, ksize=(1, 20, 20, 1), strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC")

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_0')(x)
    print(x.shape)

    x = conv_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(x)
    print(x.shape)

    x = tf.concat([x, x_80_40_256], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(x)
    print(x.shape)

    x = tf.concat([x, x_160_80_128], axis=-1)

    x = conv_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'a', strides=(1, 1), include_batchnorm=include_batchnorm)
    x = identity_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'b', include_batchnorm=include_batchnorm)
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(x)
    print(x.shape)
    x = tf.concat([x, x_320_160_64], axis=-1)

    x = conv_block(x, 3, [output_dim, output_dim, output_dim], stage=9, block=prefix + 'a', strides=(1, 1), activation=False)
    output = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=9, block=prefix + 'b', activation=False)
    print(output.shape)

    return input_data, output, global_feat


def UNet61_8s(input_shape, output_dim, include_batchnorm=False, batchnorm_axis=-1, prefix=''):
    """produces an hourglass FCN network, adapted to CoRL submission size.

    Regarding shapes, look at: https://www.tensorflow.org/api_docs/python/tf/keras/Input [excludes batch size]
    Here are the shape patterns, where I print shapes for the input, and after each conv_block or Conv2D call.

    Attention:
    (None, 320, 320, 6) (input shape)
    (None, 320, 320, 64)
    (None, 320, 320, 64)
    (None, 160, 160, 128)
    (None, 80, 80, 256)
    (None, 40, 40, 512)
    (None, 40, 40, 256)
    (None, 80, 80, 128)
    (None, 160, 160, 64)
    (None, 320, 320, 1)

    Transport, key module
    (None, 384, 224, 6) (input shape)
    (None, 384, 224, 64)
    (None, 384, 224, 64)
    (None, 192, 112, 128)
    (None, 96, 56, 256)
    (None, 48, 28, 512)
    (None, 48, 28, 256)
    (None, 96, 56, 128)
    (None, 192, 112, 64)
    (None, 384, 224, 3)

    Transport, query module, assumes cropping beforehand.
    (None, 64, 64, 6) (input shape)
    (None, 64, 64, 64)
    (None, 64, 64, 64)
    (None, 32, 32, 128)
    (None, 16, 16, 256)
    (None, 8, 8, 512)
    (None, 8, 8, 256)
    (None, 16, 16, 128)
    (None, 32, 32, 64)
    (None, 64, 64, 3)

    Here I ignore output after identity blocks, which produce tensors of the same size.

    Parameters
    ----------
    :input_shape: a tuple that specifies the shape for tf.keras.layers.Input. By default,
        it's (None,320,320,6) for Attention, (None,384,224,6) for Transport. The input is
        a tuple, not a tensor.
    :output_dim: a single scalar, which produces the number of channels of the output.
        For example, if it's set to 3, then if we pass kernels of size (None,64,64,6),
        the default, we get an output tensor of size (None,64,64,3).
    """
    input_data = tf.keras.layers.Input(shape=input_shape)
    print(input_data.shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', name=prefix + 'conv1')(input_data)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=prefix + 'bn_conv1')(x)
    x = tf.keras.layers.ReLU()(x)
    print(x.shape)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1), short_cut=False)
    x_320_320_64 = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b', short_cut=False)
    x_320_320_64 = identity_block(x_320_320_64, 3, [64, 64, 64], stage=2, block=prefix + 'c', short_cut=False)
    print(x_320_320_64.shape)

    x_160_160_128 = conv_block(x_320_320_64, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2), short_cut=False)
    x_160_160_128 = identity_block(x_160_160_128, 3, [128, 128, 128], stage=3, block=prefix + 'b', short_cut=False)
    x_160_160_128 = identity_block(x_160_160_128, 3, [128, 128, 128], stage=3, block=prefix + 'c', short_cut=False)
    print(x_160_160_128.shape)

    x_80_80_256 = conv_block(x_160_160_128, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
    x_80_80_256 = identity_block(x_80_80_256, 3, [256, 256, 256], stage=4, block=prefix + 'b')
    x_80_80_256 = identity_block(x_80_80_256, 3, [256, 256, 256], stage=4, block=prefix + 'c')
    print(x_80_80_256.shape)

    x_40_40_512 = conv_block(x_80_80_256, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
    x_40_40_512 = identity_block(x_40_40_512, 3, [512, 512, 512], stage=5, block=prefix + 'b')
    x_40_40_512 = identity_block(x_40_40_512, 3, [512, 512, 512], stage=5, block=prefix + 'c')
    print(x_40_40_512.shape)

    x_20_20_512 = conv_block(x_40_40_512, 3, [512, 512, 512], stage=6, block=prefix + 'a', strides=(2, 2))
    x_20_20_512 = identity_block(x_20_20_512, 3, [512, 512, 512], stage=6, block=prefix + 'b')
    x_20_20_512 = identity_block(x_20_20_512, 3, [512, 512, 512], stage=6, block=prefix + 'c')
    print(x_20_20_512.shape)

    global_feat = tf.nn.avg_pool(x_20_20_512, ksize=(1, 20, 20, 1), strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC")

    x_40_40_512_U = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_0')(x_20_20_512)
    print(x_40_40_512_U.shape)

    x_40_40_256_U = conv_block(x_40_40_512_U, 3, [256, 256, 256], stage=7, block=prefix + 'a', strides=(1, 1))
    x_40_40_256_U = identity_block(x_40_40_256_U, 3, [256, 256, 256], stage=7, block=prefix + 'b')
    print(x_40_40_256_U.shape)

    x_80_80_256_U = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(x_40_40_256_U)
    print(x_80_80_256_U.shape)

    x = tf.concat([x_80_80_256_U, x_80_80_256], axis=-1)

    x = conv_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'a', strides=(1, 1), short_cut=False)
    x = identity_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'b', short_cut=False)
    x = identity_block(x, 3, [128, 128, 128], stage=8, block=prefix + 'c', short_cut=False)
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(x)
    print(x.shape)

    x = tf.concat([x, x_160_160_128], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=9, block=prefix + 'a', strides=(1, 1), short_cut=False)
    x = identity_block(x, 3, [128, 128, 128], stage=9, block=prefix + 'b', short_cut=False)
    x = identity_block(x, 3, [64, 64, 64], stage=9, block=prefix + 'c', short_cut=False)
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(x)
    print(x.shape)
    x = tf.concat([x, x_320_320_64], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'a', strides=(1, 1), short_cut=False)
    x = conv_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'b', strides=(1, 1), short_cut=False)
    x = identity_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'c', short_cut=False)

    x = conv_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'a', strides=(1, 1), activation=False, short_cut=False)
    x = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'b', activation=False, short_cut=False)
    output = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'c', activation=False, short_cut=False)
    print(output.shape)

    return input_data, output, global_feat


def UNet47_8s(input_shape, output_dim, include_batchnorm=False, batchnorm_axis=-1, prefix=''):
    """produces an hourglass FCN network, adapted to CoRL submission size.

    Regarding shapes, look at: https://www.tensorflow.org/api_docs/python/tf/keras/Input [excludes batch size]
    Here are the shape patterns, where I print shapes for the input, and after each conv_block or Conv2D call.

    Attention:
    (None, 320, 320, 6) (input shape)
    (None, 320, 320, 64)
    (None, 320, 320, 64)
    (None, 160, 160, 128)
    (None, 80, 80, 256)
    (None, 40, 40, 512)
    (None, 40, 40, 256)
    (None, 80, 80, 128)
    (None, 160, 160, 64)
    (None, 320, 320, 1)

    Transport, key module
    (None, 384, 224, 6) (input shape)
    (None, 384, 224, 64)
    (None, 384, 224, 64)
    (None, 192, 112, 128)
    (None, 96, 56, 256)
    (None, 48, 28, 512)
    (None, 48, 28, 256)
    (None, 96, 56, 128)
    (None, 192, 112, 64)
    (None, 384, 224, 3)

    Transport, query module, assumes cropping beforehand.
    (None, 64, 64, 6) (input shape)
    (None, 64, 64, 64)
    (None, 64, 64, 64)
    (None, 32, 32, 128)
    (None, 16, 16, 256)
    (None, 8, 8, 512)
    (None, 8, 8, 256)
    (None, 16, 16, 128)
    (None, 32, 32, 64)
    (None, 64, 64, 3)

    Here I ignore output after identity blocks, which produce tensors of the same size.

    Parameters
    ----------
    :input_shape: a tuple that specifies the shape for tf.keras.layers.Input. By default,
        it's (None,320,320,6) for Attention, (None,384,224,6) for Transport. The input is
        a tuple, not a tensor.
    :output_dim: a single scalar, which produces the number of channels of the output.
        For example, if it's set to 3, then if we pass kernels of size (None,64,64,6),
        the default, we get an output tensor of size (None,64,64,3).
    """
    input_data = tf.keras.layers.Input(shape=input_shape)
    print(input_data.shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', name=prefix + 'conv1')(input_data)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=prefix + 'bn_conv1')(x)
    x = tf.keras.layers.ReLU()(x)
    print(x.shape)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1), short_cut=False)
    x_320_320_64 = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b', short_cut=False)
    x_320_320_64 = identity_block(x_320_320_64, 3, [64, 64, 64], stage=2, block=prefix + 'c')
    print(x_320_320_64.shape)

    x_160_160_128 = conv_block(x_320_320_64, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2), short_cut=False)
    x_160_160_128 = identity_block(x_160_160_128, 3, [128, 128, 128], stage=3, block=prefix + 'b', short_cut=False)
    x_160_160_128 = identity_block(x_160_160_128, 3, [128, 128, 128], stage=3, block=prefix + 'c')
    print(x_160_160_128.shape)

    x_80_80_256 = conv_block(x_160_160_128, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
    x_80_80_256 = identity_block(x_80_80_256, 3, [256, 256, 256], stage=4, block=prefix + 'c')
    print(x_80_80_256.shape)

    x_40_40_512 = conv_block(x_80_80_256, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
    x_40_40_512 = identity_block(x_40_40_512, 3, [512, 512, 512], stage=5, block=prefix + 'c')
    print(x_40_40_512.shape)

    x_20_20_512 = conv_block(x_40_40_512, 3, [512, 512, 512], stage=6, block=prefix + 'a', strides=(2, 2))
    x_20_20_512 = identity_block(x_20_20_512, 3, [512, 512, 512], stage=6, block=prefix + 'c')
    print(x_20_20_512.shape)

    global_feat = tf.nn.avg_pool(x_20_20_512, ksize=(1, 20, 20, 1), strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC")

    x_40_40_512_U = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_0')(x_20_20_512)
    print(x_40_40_512_U.shape)

    x_40_40_256_U = conv_block(x_40_40_512_U, 3, [256, 256, 256], stage=7, block=prefix + 'a', strides=(1, 1))
    x_40_40_256_U = identity_block(x_40_40_256_U, 3, [256, 256, 256], stage=7, block=prefix + 'b')
    print(x_40_40_256_U.shape)

    x_80_80_256_U = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(x_40_40_256_U)
    print(x_80_80_256_U.shape)

    x = tf.concat([x_80_80_256_U, x_80_80_256], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=8, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 128], stage=8, block=prefix + 'c')
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(x)
    print(x.shape)

    x = tf.concat([x, x_160_160_128], axis=-1)

    x = conv_block(x, 3, [64, 64, 64], stage=9, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 64], stage=9, block=prefix + 'c')
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(x)
    print(x.shape)
    x = tf.concat([x, x_320_320_64], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'a', strides=(1, 1))
    x = conv_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'b', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 128], stage=10, block=prefix + 'c')

    x = conv_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'a', strides=(1, 1), activation=False, short_cut=False)
    x = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'b', activation=False, short_cut=False)
    output = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=11, block=prefix + 'c', activation=False, short_cut=False)
    print(output.shape)

    return input_data, output, global_feat


def UNet43_8s_aff(input_shape, output_dim, include_batchnorm=False, batchnorm_axis=-1, prefix='', cutoff_early=False):
    """produces an hourglass FCN network, adapted to CoRL submission size.

    Regarding shapes, look at: https://www.tensorflow.org/api_docs/python/tf/keras/Input [excludes batch size]
    Here are the shape patterns, where I print shapes for the input, and after each conv_block or Conv2D call.

    Attention:
    (None, 320, 320, 6) (input shape)
    (None, 320, 320, 64)
    (None, 320, 320, 64)
    (None, 160, 160, 128)
    (None, 80, 80, 256)
    (None, 40, 40, 512)
    (None, 40, 40, 256)
    (None, 80, 80, 128)
    (None, 160, 160, 64)
    (None, 320, 320, 1)

    Transport, key module
    (None, 384, 224, 6) (input shape)
    (None, 384, 224, 64)
    (None, 384, 224, 64)
    (None, 192, 112, 128)
    (None, 96, 56, 256)
    (None, 48, 28, 512)
    (None, 48, 28, 256)
    (None, 96, 56, 128)
    (None, 192, 112, 64)
    (None, 384, 224, 3)

    Transport, query module, assumes cropping beforehand.
    (None, 64, 64, 6) (input shape)
    (None, 64, 64, 64)
    (None, 64, 64, 64)
    (None, 32, 32, 128)
    (None, 16, 16, 256)
    (None, 8, 8, 512)
    (None, 8, 8, 256)
    (None, 16, 16, 128)
    (None, 32, 32, 64)
    (None, 64, 64, 3)

    Here I ignore output after identity blocks, which produce tensors of the same size.

    Parameters
    ----------
    :input_shape: a tuple that specifies the shape for tf.keras.layers.Input. By default,
        it's (None,320,320,6) for Attention, (None,384,224,6) for Transport. The input is
        a tuple, not a tensor.
    :output_dim: a single scalar, which produces the number of channels of the output.
        For example, if it's set to 3, then if we pass kernels of size (None,64,64,6),
        the default, we get an output tensor of size (None,64,64,3).
    """
    input_data = tf.keras.layers.Input(shape=input_shape)
    print(input_data.shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', name=prefix + 'conv1')(input_data)
    if include_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis, name=prefix + 'bn_conv1')(x)
    x = tf.keras.layers.ReLU()(x)
    print(x.shape)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
    x_320_160_64 = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')
    print(x.shape)

    x = conv_block(x_320_160_64, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2))
    x_160_80_128 = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b')
    print(x.shape)

    x = conv_block(x_160_80_128, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
    x_80_40_256 = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b')
    print(x_80_40_256.shape)

    x = conv_block(x_80_40_256, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
    x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b')
    print(x.shape)

    x = conv_block(x, 3, [512, 512, 512], stage=17, block=prefix + 'a', strides=(2, 2))
    x = identity_block(x, 3, [512, 512, 512], stage=17, block=prefix + 'b')
    print(x.shape)

    global_feat = tf.nn.avg_pool(x, ksize=(1, 20, 10, 1), strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC")

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_0')(x)
    print(x.shape)

    x = conv_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b')
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(x)
    print(x.shape)

    x = tf.concat([x, x_80_40_256], axis=-1)

    x = conv_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b')
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(x)
    print(x.shape)

    x = tf.concat([x, x_160_80_128], axis=-1)

    x = conv_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'a', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 256], stage=8, block=prefix + 'b')
    print(x.shape)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(x)
    print(x.shape)
    x = tf.concat([x, x_320_160_64], axis=-1)

    x = conv_block(x, 3, [output_dim, output_dim, output_dim], stage=9, block=prefix + 'a', strides=(1, 1), activation=False)
    output = identity_block(x, 3, [output_dim, output_dim, output_dim], stage=9, block=prefix + 'b', activation=False)
    print(output.shape)

    return input_data, output, global_feat

