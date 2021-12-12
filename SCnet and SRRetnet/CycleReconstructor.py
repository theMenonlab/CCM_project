import tensorflow as tf
import tensorflow_addons as tfa

from CycleReconstructorBlocks import *


def cycle_reconstructor(params):
    filters = params['filters']
    kernels = params['kernels']
    dilation_rate = params['dilation_rate']
    dropout = params['dropout']
    res_depth = params['res_depth']
    input_shape = params['input_shape']
    output_shape = params['output_shape']
    num_channels = params['num_channels']
    channel_format = params['channel_format']
    skip = params['skip']

    if params['norm'] == 'instance': norm_layer = tfa.layers.InstanceNormalization
    elif params['norm'] == 'batch':  norm_layer = tf.keras.layers.BatchNormalization
    else: raise ValueError('Normalization must be one of "batch" or "instance".')

    if params['activation'] == 'relu': activation = tf.keras.layers.ReLU
    else: raise ValueError('Activation must be one of "relu" or added to CycleReconstructor.py under line 23.')

    return cycle_reconstructor_even(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels, input_shape, skip)


def cycle_reconstructor_even(filters={'down':[64,64,64,64], 'up':[64]}, 
                                          kernels=[3,3], 
                                          dilation_rate=1, 
                                          dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                          res_depth={'down':1, 'bottom':0, 'up':1},
                                          norm_layer=tfa.layers.InstanceNormalization, 
                                          activation=tf.keras.layers.ReLU, 
                                          channel_format='NCHW', num_channels=3, input_shape=(128,128), skip=False):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    skips = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    
    if kernels[0] == 7: top_layer_padding = 3
    elif kernels[0] == 5: top_layer_padding = 2
    elif kernels[0] == 3: top_layer_padding = 1
    if input_shape == (447,447):
      top_layer_padding = [[top_layer_padding,top_layer_padding-1], [top_layer_padding,top_layer_padding-1]]
    else:
      top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
    else:
      input = tf.keras.layers.Input((input_shape[0], input_shape[1], num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())

    # Down
    d1 = conv_norm_act(top, down_filters[0], kernels[1], (1,1), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    skips.append(d1)

    down = d1
    for i in range(1, len(down_filters)):
        down = conv_norm_act(down, down_filters[i], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        if i < len(down_filters) - 1:
            skips.append(down)

    # down = non_local(down, down_filters[3] // 2, channel_format=channel_format, mode='emb')

    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Up
    up = r
    for i in range(len(up_filters)):
        # Extra padding for weird shape input/output
        if i == len(up_filters) - 1: 
            if input_shape == (338,338):
                up = reflect_padding(up, [[1,0], [1,0]], channel_format=channel_format)
            elif input_shape == (468,468):
                up = reflect_padding(up, [[1,1], [1,1]], channel_format=channel_format)

        up = convtranspose_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        if skip: up = skip_connection(True, up, skips[-(i+1)], concat_axis, up_filters[i], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    up = conv_norm_act(up, 64, kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())

    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    model = tf.keras.models.Model(input, top, name='cycle_reconstructor_even')
    return model
