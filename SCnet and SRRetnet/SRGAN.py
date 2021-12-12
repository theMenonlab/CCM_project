import tensorflow as tf


## Generator
def srgan_gen(input_shape=(128,128), channel_format='NCHW', num_channels=1, num_residuals=16):
    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
        norm_axis = 1
    else:
        input = tf.keras.layers.Input((input_shape[0], input_shape[1], num_channels))
        norm_axis = -1

    l1 = tf.keras.layers.Convolution2D(64, (9,9), strides=1, padding='same')(input)
    l1 = tf.keras.layers.PReLU()(l1)
    
    x = l1
    for b in range(num_residuals):
        x = sr_residual_block(x, norm_axis)

    x = tf.keras.layers.Convolution2D(64, (3,3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=norm_axis)(x)
    x = x + l1

    x = tf.keras.layers.Convolution2D(256, (3,3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=norm_axis)(x)
    x = tf.keras.layers.PReLU()(x)
    
    x = tf.keras.layers.Convolution2D(256, (3,3), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=norm_axis)(x)
    x = tf.keras.layers.PReLU()(x)

    out = tf.keras.layers.Convolution2D(num_channels, (9,9), strides=1, padding='same')(x)
    # out = tf.keras.activations.sigmoid(out)

    model = tf.keras.models.Model(input, out, name='srgan_gen')
    return model


def sr_residual_block(input, norm_axis=1):
    l1 = tf.keras.layers.Convolution2D(64, (3,3), strides=1, padding='same')(input)
    l1 = tf.keras.layers.BatchNormalization(axis=norm_axis)(l1)
    l1 = tf.keras.layers.PReLU()(l1)

    l2 = tf.keras.layers.Convolution2D(64, (3,3), strides=1, padding='same')(l1)
    l2 = tf.keras.layers.BatchNormalization(axis=norm_axis)(l2)
    return input + l2


## Discriminator
def srgan_disc(input_shape=(128,128), channel_format='NCHW', num_channels=1):
    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
        norm_axis = 1
    else:
        input = tf.keras.layers.Input((input_shape[0], input_shape[1], num_channels))
        norm_axis = -1
    
    l1 = sr_disc_block(input, 64, 1, False, norm_axis=norm_axis)
    l2 = sr_disc_block(l1, 64, 2, norm_axis=norm_axis)

    l3 = sr_disc_block(l2, 128, 1, norm_axis=norm_axis)
    l4 = sr_disc_block(l3, 128, 2, norm_axis=norm_axis)

    l5 = sr_disc_block(l4, 256, 1, norm_axis=norm_axis)
    l6 = sr_disc_block(l5, 256, 2, norm_axis=norm_axis)

    l7 = sr_disc_block(l6, 512, 1, norm_axis=norm_axis)
    l8 = sr_disc_block(l7, 512, 2, norm_axis=norm_axis)

    out = tf.keras.layers.Flatten()(l8)
    out = tf.keras.layers.Dense(1024)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.2)(out)
    out = tf.keras.layers.Dense(1)(out)
    out = tf.keras.activations.sigmoid(out)

    model = tf.keras.models.Model(input, out, name='srgan_disc')
    return model


def sr_disc_block(input, num_filters, strides=1, apply_norm=True, norm_axis=1):
    x = tf.keras.layers.Convolution2D(num_filters, (3,3), strides=strides, padding='same')(input)
    if apply_norm: 
        x = tf.keras.layers.BatchNormalization(axis=norm_axis)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x