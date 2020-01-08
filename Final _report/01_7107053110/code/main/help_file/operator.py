import tensorflow as tf

def lrelu(x, a):
    with tf.variable_scope("lrleu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def Conv(batch_input, output_channel, stride, pad, name):
    with tf.variable_scope("Conv_{}".format(str(name))):
        # [batch, in_height, in_width, in_channels]
        input_channel = batch_input.get_shape()[3]
        # [filter_width, filter_height, in_channels, out_channels]
        weights = tf.get_variable(name="filter_{}".format(str(name)), shape=[4, 4, input_channel, output_channel],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        conv = tf.nn.conv2d(input=batch_input, filter=weights, strides=[1, stride, stride, 1], padding=pad)
        return conv


def Deconv(batch_input, output_channel, stride, name):
    with tf.variable_scope("Deconv_{}".format(str(name))):
        batch_size, input_height, input_width, input_channel = [int(i) for i in batch_input.get_shape()]
        weights = tf.get_variable(name="filter_{}".format(str(name)), shape=[4, 4, output_channel, input_channel],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input=batch_input, filter=weights,
                                        output_shape=[batch_size, input_height * stride, input_width * stride, output_channel],
                                        strides=[1, stride, stride, 1], padding="SAME")
        return deconv


def Batchnorm(batch_input, name):
    with tf.variable_scope("Batchnorm_{}".format(str(name))):
        input = tf.identity(batch_input)
        channel = input.get_shape()[3]
        offset = tf.get_variable(name="offset_{}".format(str(name)), shape=[channel], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable(name="scale_{}".format(str(name)), shape=[channel], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        norm = tf.nn.batch_normalization(x=input, mean=mean, variance=variance, offset=offset, scale=scale,
                                         variance_epsilon=variance_epsilon)
        return norm


