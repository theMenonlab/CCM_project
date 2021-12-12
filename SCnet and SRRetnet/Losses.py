import tensorflow as tf


def dist_compute_metrics(x, fy):
    nhwc_x = tf.transpose(x, [0, 2, 3, 1])
    nhwc_fy = tf.transpose(fy, [0, 2, 3, 1])
    f_ssim = tf.nn.compute_average_loss(tf.image.ssim(nhwc_x, nhwc_fy, 1.))
    f_psnr = tf.nn.compute_average_loss(tf.image.psnr(nhwc_x, nhwc_fy, 1.))
    return f_ssim, f_psnr

def dist_mae_loss(y_true, y_pred, axis=[1,2,3]):
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.abs(y_true - y_pred), axis=axis))

def dist_mse_loss(y_true, y_pred, axis=[1,2,3]):
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=axis))

def dist_ls_gan_loss_gen(disc_out_fake):
    """
    (D_x(F(y)) - 1)^2 or (D_y(G(x)) - 1)^2

    disc_out_fake: D_x(F(y)) or D_y(G(x))
    """
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 1), axis=[1,2,3]))

def dist_ls_gan_loss_disc(disc_out_real, disc_out_fake):
    """
    (D_x(x) - 1)^2 + D_x(F(y))^2 or (D_y(y) - 1)^2 + D_y(G(x))^2

    disc_out_real: D_x(x), and disc_x_out_fake: D_x(F(y)) 
    or disc_out_real: D_y(y) and disc_x_out_fake: D_y(G(x))
    """
    disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 0.9), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0), axis=[1,2,3])
    # disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, tf.random.uniform((), minval=0.7, maxval=1.)), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, tf.random.uniform((), minval=0., maxval=0.3)), axis=[1,2,3])
    # disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 1), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0), axis=[1,2,3])
    return 0.5 * tf.nn.compute_average_loss(disc_loss)


def dist_la_gan_loss_gen(disc_out_fake):
    """
    |D_x(F(y)) - 1| or |D_y(G(x)) - 1|

    disc_out_fake: D_x(F(y)) or D_y(G(x))
    """
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.abs(disc_out_fake - 1), axis=[1,2,3]))

def dist_la_gan_loss_disc(disc_out_real, disc_out_fake):
    """
    |D_x(x) - 1| + |D_x(F(y))| or |D_y(y) - 1| + |D_y(G(x))|

    disc_out_real: D_x(x), and disc_x_out_fake: D_x(F(y)) 
    or disc_out_real: D_y(y) and disc_x_out_fake: D_y(G(x))
    """
    disc_loss = tf.reduce_mean(tf.abs(disc_out_real - 0.9), axis=[1,2,3]) + tf.reduce_mean(tf.abs(disc_out_fake), axis=[1,2,3])
    return 0.5 * tf.nn.compute_average_loss(disc_loss)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
def dist_gan_loss_gen(disc_out_fake):
    return tf.nn.compute_average_loss(cross_entropy(tf.ones_like(disc_out_fake), disc_out_fake))

def dist_gan_loss_disc(disc_out_real, disc_out_fake):
    real_loss = cross_entropy(tf.ones_like(disc_out_real), disc_out_real)
    fake_loss = cross_entropy(tf.zeros_like(disc_out_fake), disc_out_fake)
    total_loss = real_loss + fake_loss
    return tf.nn.compute_average_loss(total_loss)


def reconstruction_loss(real, gen, loss_func):
    """
    real: x and gen: F(y)
    or real: y and gen: G(x)
    """
    return loss_func(real, gen)


def cycle_loss(real, cycled, loss_func, LAMBDA):
    """
    real: x and cycled: F(G(x))
    or real: y and cycled: G(F(y))
    """
    return LAMBDA * loss_func(real, cycled)
