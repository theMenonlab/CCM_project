import tensorflow as tf
import tensorflow_addons as tfa


def conv_norm(input, filters, kernel_size=3, strides=(1,1), padding='same', dilation_rate=1,
                  norm_layer=tfa.layers.InstanceNormalization):
    dk = tf.keras.layers.Convolution2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(input)
    dk = norm_layer(axis=1)(dk)
    return dk


def conv_norm_act(input, filters, kernel_size=3, strides=(1,1), padding='same', dilation_rate=1,
                  norm_layer=tfa.layers.InstanceNormalization, activation=tf.keras.layers.ReLU()):
    dk = tf.keras.layers.Convolution2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(input)
    if norm_layer is not None:
        dk = norm_layer(axis=1)(dk)
    dk = activation(dk)
    return dk


def reflect_padding(input, padding, channel_format='NCHW'):
    if channel_format == 'NCHW':
        return tf.pad(input, [[0,0], [0,0], padding[0], padding[1]], 'REFLECT')
    else: 
        return tf.pad(input, [[0,0], padding[0], padding[1], [0,0]], 'REFLECT')


def residual_block(input, filters, kernel_size=3, padding=[[1,1], [1,1]], dilation_rate=1,
                   norm_layer=tfa.layers.InstanceNormalization, activation=tf.keras.layers.ReLU(), channel_format='NCHW'):

    r_1 = reflect_padding(input, padding, channel_format)
    r_1 = conv_norm_act(r_1, filters, kernel_size, (1,1), 'valid', dilation_rate, norm_layer, activation)

    r_2 = reflect_padding(r_1, padding, channel_format)
    r_2 = tf.keras.layers.Convolution2D(filters, kernel_size, padding='valid', dilation_rate=dilation_rate)(r_2)
    r_2 = norm_layer(axis=1)(r_2)

    x = tf.keras.layers.Add()([input, r_2])
    x = activation(x)
    return x


def convtranspose_norm_act(input, filters, kernel_size=3, dilation_rate=1, strides=(2,2),
                           norm_layer=tfa.layers.InstanceNormalization, activation=tf.keras.layers.ReLU()):
    uk = tf.keras.layers.Convolution2DTranspose(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(input)
    uk = norm_layer(axis=1)(uk)
    uk = activation(uk)
    return uk


# Skip Connections
def get_crop_shape(skip, input, concat_axis):
    cw = skip.get_shape()[3] - input.get_shape()[3] if concat_axis == 1 else skip.get_shape()[2] - input.get_shape()[2]
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    ch = skip.get_shape()[2] - input.get_shape()[2] if concat_axis == 1 else skip.get_shape()[1] - input.get_shape()[1]
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def crop_or_pad(skip, input, concat_axis):
    if concat_axis == 1: axes = [2,3]
    else: axes = [1,2]
    
    if skip.shape[axes[0]] == input.shape[axes[0]] and skip.shape[axes[1]] == input.shape[axes[1]]:
        return input
    elif skip.shape[axes[0]] > input.shape[axes[0]] or skip.shape[axes[1]] > input.shape[axes[1]]:
        return tf.keras.layers.Cropping2D((get_crop_shape(skip, input, concat_axis)))(skip)
    else:
        pad_shape = get_crop_shape(input, skip, concat_axis)
        padding = [[0,0], [0,0], [0,0], [0,0]]
        padding[axes[0]] = [pad_shape[0][0], pad_shape[0][1]]
        padding[axes[1]] = [pad_shape[1][0], pad_shape[1][1]]
        return tf.pad(skip, padding)


def skip_connection(reshape, input, skip, concat_axis, filters, kernel_size, dilation_rate, norm_layer=tfa.layers.InstanceNormalization, activation=tf.keras.layers.ReLU()):
    if reshape:
        x = crop_or_pad(skip, input, concat_axis)
    else:
        x = skip
    x = tf.keras.layers.Concatenate(axis=concat_axis)([x, input])
    x = conv_norm_act(x, filters, kernel_size, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation)
    return x


# Non local
def non_local(input, filters, channel_format, mode='emb'):
    channel = 1 if channel_format == 'NCHW' else -1
    num_input_channels = input.shape[channel]

    # Gaussian
    if mode == 'gaussian':
        theta = input
        phi = input
        if channel == 1:
            theta = tf.keras.layers.Permute([2, 3, 1])(theta)
            phi = tf.keras.layers.Permute([2, 3, 1])(phi)

        theta = tf.keras.layers.Reshape([-1, filters])(theta) # [BS, HW, C]
        phi = tf.keras.layers.Reshape([-1, filters])(phi) # [BS, HW, C]

        # Attention map
        s = tf.matmul(theta, phi, transpose_b=True) # [BS, HW, HW]
        s = tf.nn.softmax(s)

    # Embedded
    elif mode == 'emb':
        theta = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
        phi = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
        phi = tf.keras.layers.MaxPooling2D()(phi)  # GB

        if channel == 1:
            theta = tf.keras.layers.Permute([2, 3, 1])(theta)
            phi = tf.keras.layers.Permute([2, 3, 1])(phi)

        theta = tf.keras.layers.Reshape([-1, filters])(theta) # [BS, HW, C]
        phi = tf.keras.layers.Reshape([-1, filters])(phi) # [BS, HW, C]

        # Attention map
        s = tf.matmul(theta, phi, transpose_b=True) # [BS, HW, HW]
        s = tf.nn.softmax(s)

    g = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
    if mode == 'emb': g = tf.keras.layers.MaxPooling2D()(g)  # GB
    if channel == 1: g = tf.keras.layers.Permute([2, 3, 1])(g)
    g = tf.keras.layers.Reshape([-1, filters])(g) # [BS, HW, C]

    o = tf.matmul(s, g) # [BS, HW, C]
    if channel == 1:
        o = tf.keras.layers.Reshape([input.shape[2], input.shape[3], filters])(o)
        o = tf.keras.layers.Permute([3, 1, 2])(o) # [BS, H, W, C]
    else:
        o = tf.keras.layers.Reshape([input.shape[1], input.shape[2], filters])(o) # [BS, H, W, C]
    o = tf.keras.layers.Conv2D(num_input_channels, kernel_size=1, padding='same', kernel_initializer='zeros')(o)
    o = tf.keras.layers.BatchNormalization(axis=channel)(o) # Section 4.1

    x = o + input
    return x
