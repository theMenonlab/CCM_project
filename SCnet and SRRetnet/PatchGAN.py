import tensorflow as tf
import tensorflow_addons as tfa


# Discriminator
def patch_gan(input_shape, num_channels=3, norm_layer='instance', channel_format='NCHW'):
    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
    else:
        input = tf.keras.layers.Input((input_shape[0], input_shape[1], num_channels))
    
    if norm_layer == 'instance': norm_layer = tfa.layers.InstanceNormalization
    elif norm_layer == 'batch':  norm_layer = tf.keras.layers.BatchNormalization
    else: raise ValueError('Normalization must be one of "batch" or "instance".')

    return patch_gan_70x70(input, norm_layer)

    
def patch_gan_70x70(input, norm_layer=tfa.layers.InstanceNormalization):
    x = input
    # x = tf.keras.layers.GaussianNoise(0.1)(x)
    d1 = patch_gan_down_block(x, 64, norm_layer=norm_layer, apply_norm=False)
    d1 = tf.keras.layers.Dropout(0.3)(d1)

    d2 = patch_gan_down_block(d1, 128, norm_layer=norm_layer, apply_norm=True)
    d2 = tf.keras.layers.Dropout(0.3)(d2)

    d3 = patch_gan_down_block(d2, 256, norm_layer=norm_layer, apply_norm=True)
    d3 = tf.keras.layers.Dropout(0.3)(d3)

    d4 = tf.keras.layers.ZeroPadding2D()(d3)
    d4 = patch_gan_down_block(d4, 512, strides=1, norm_layer=norm_layer, apply_norm=True)
    d4 = tf.keras.layers.Dropout(0.3)(d4)
    
    out = tf.keras.layers.ZeroPadding2D()(d4)
    out = tf.keras.layers.Conv2D(1, 4, strides=1)(out)
    out = tf.keras.activations.sigmoid(out)

    return tf.keras.models.Model(input, out, name='patch_gan_70x70')


def patch_gan_down_block(input, filters, strides=2, norm_layer=tfa.layers.InstanceNormalization, apply_norm=True):
    dk = tf.keras.layers.Conv2D(filters, 4, strides=strides, padding='SAME')(input)
    if apply_norm: 
        dk = norm_layer(axis=1)(dk)
    dk = tf.keras.layers.LeakyReLU(0.2)(dk)
    return dk