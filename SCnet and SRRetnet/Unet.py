import tensorflow as tf

def Unet(starting_filters=32, input_shape=(240,320), num_channels=3):
    # input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
    input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))

    l2 = conv_block(input, starting_filters)
    l2 = dense_block(l2, starting_filters)

    # l3 = tf.keras.layers.MaxPooling2D()(l2)
    l3 = conv_block(l2, starting_filters*2, strides=2)
    l3 = dense_block(l3, starting_filters*2)

    # l4 = tf.keras.layers.MaxPooling2D()(l3)
    l4 = conv_block(l3, starting_filters*4, strides=2)
    l4 = dense_block(l4, starting_filters*4)

    # l5 = tf.keras.layers.MaxPooling2D()(l4)
    l5 = conv_block(l4, starting_filters*8, strides=2)
    l5 = dense_block(l5, starting_filters*8)

    # # l6 = tf.keras.layers.MaxPooling2D()(l5)
    # l6 = conv_block(l6, starting_filters*16, strides=2)
    # l6 = dense_block(l6, starting_filters*16)

    # r5 = tf.keras.layers.Conv2DTranspose(starting_filters*8, (3,3), (2,2), padding='same')(l6)
    # m5 = tf.keras.layers.Concatenate(axis=1)([l5, r5])
    # # m5 = tf.keras.layers.Concatenate(axis=1)([m5, r5])
    # r5 = dense_block(m5, starting_filters*8)

    r4 = tf.keras.layers.Conv2DTranspose(starting_filters*4, (3,3), (2,2), padding='same')(l5)
    if input_shape == (468,468): r4 = tf.pad(r4, [[0,0], [0,0], [1,0], [1,0]])
    m4 = tf.keras.layers.Concatenate(axis=1)([l4, r4])
    m4 = conv_block(m4, starting_filters*4)
    r4 = dense_block(m4, starting_filters*4)

    r3 = tf.keras.layers.Conv2DTranspose(starting_filters*2, (3,3), (2,2), padding='same')(r4)
    if input_shape == (338,338): r3 = tf.pad(r3, [[0,0], [0,0], [1,0], [1,0]])
    m3 = tf.keras.layers.Concatenate(axis=1)([l3, r3])
    # m3 = tf.keras.layers.Concatenate(axis=1)([m3, r3])
    m3 = conv_block(m3, starting_filters*2)
    r3 = dense_block(m3, starting_filters*2)

    r2 = tf.keras.layers.Conv2DTranspose(starting_filters, (3,3), (2,2), padding='same')(r3)
    m2 = tf.keras.layers.Concatenate(axis=1)([l2, r2])
    # m2 = tf.keras.layers.Concatenate()([m2, r2])
    m2 = conv_block(m2, starting_filters)
    r2 = dense_block(m2, starting_filters)

    r1 = tf.keras.layers.Conv2D(num_channels, (1,1), activation='sigmoid', padding='same')(r2)

    return tf.keras.models.Model(input, r1)


def conv_block(input, filters, strides=1):
    x = tf.keras.layers.Conv2D(filters, (3,3), strides=strides, padding='same')(input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    return x


def dense_block(input, filters):
    l1 = tf.keras.layers.Conv2D(filters, (3,3), padding='same')(input)
    # x = tf.keras.layers.BatchNormalization()(x)
    l1 = tf.keras.layers.ReLU()(l1)
    l2 = conv_block(l1, filters)
    return l1 + l2