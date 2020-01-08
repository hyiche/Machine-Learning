from .operator import *
import tensorflow as tf

# def generator(L_img, Mos_gray_list, ngf=64):
#     with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
#         gen_images_list = []
#
#         with tf.variable_scope("encoder_1"):
#             input_ = tf.concat([L_img, Mos_gray_list[0]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_1"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#
#         with tf.variable_scope("encoder_2"):
#             input_ = tf.concat([L_img, Mos_gray_list[1], gen_images_list[-1]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_2"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#
#         with tf.variable_scope("encoder_3"):
#             input_ = tf.concat([L_img, Mos_gray_list[2], gen_images_list[-1]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_3"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#
#         with tf.variable_scope("encoder_4"):
#             input_ = tf.concat([L_img, Mos_gray_list[3], gen_images_list[-1]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_4"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#
#         with tf.variable_scope("encoder_5"):
#             input_ = tf.concat([L_img, Mos_gray_list[4], gen_images_list[-1]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_5"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#
#         with tf.variable_scope("encoder_6"):
#             input_ = tf.concat([L_img, gen_images_list[-1]], axis=3)
#             net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf)
#
#             net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 2)
#
#             net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
#             net = lrelu(x=net, a=0.2)
#             net = Batchnorm(batch_input=net, name=ngf * 4)
#
#         with tf.variable_scope("decoder_6"):
#             denet = Deconv(batch_input=net, output_channel=ngf * 2, stride=2, name=ngf * 2)
#             denet = Batchnorm(batch_input=denet, name=ngf * 2)
#
#             denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
#             denet = Batchnorm(batch_input=denet, name=ngf)
#
#             denet = Deconv(batch_input=denet, output_channel=2, stride=2, name=2)
#             denet = tf.nn.tanh(denet)
#             gen_images_list.append(denet)
#     return gen_images_list


def generator(L_img, name, Mos_gen_img=None, Mos_gray_img=None, ngf=64):
    with tf.variable_scope("generator_{}".format(str(name)), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("encoder_{}".format(str(name))):
            if Mos_gen_img is None:
                input_ = tf.concat([L_img, Mos_gray_img], axis=3)
            elif Mos_gray_img is None:
                input_ = tf.concat([L_img, Mos_gen_img], axis=3)
            else:
                input_ = tf.concat([L_img, Mos_gen_img, Mos_gray_img], axis=3)

            net = Conv(batch_input=input_, output_channel=ngf, stride=2, pad="SAME", name=ngf)
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=ngf)

            net = Conv(batch_input=net, output_channel=ngf * 2, stride=2, pad="SAME", name=ngf * 2)
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=ngf * 2)

            net = Conv(batch_input=net, output_channel=ngf * 4, stride=2, pad="SAME", name=ngf * 4)
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=ngf * 4)

            net = Conv(batch_input=net, output_channel=ngf * 8, stride=2, pad="SAME", name=str(ngf * 8)+"_1")
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=str(ngf * 8)+"_1")

            net = Conv(batch_input=net, output_channel=ngf * 8, stride=2, pad="SAME", name=str(ngf * 8)+"_2")
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=str(ngf * 8)+"_2")

            net = Conv(batch_input=net, output_channel=ngf * 8, stride=2, pad="SAME", name=str(ngf * 8)+"_3")
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=str(ngf * 8)+"_3")

            net = Conv(batch_input=net, output_channel=ngf * 8, stride=2, pad="SAME", name=str(ngf * 8)+"_4")
            net = lrelu(x=net, a=0.2)
            net = Batchnorm(batch_input=net, name=str(ngf * 8)+"_4")

        with tf.variable_scope("decoder_{}".format(str(name))):
            denet = Deconv(batch_input=net, output_channel=ngf * 8, stride=2, name=str(ngf * 8)+"_1")
            denet = Batchnorm(batch_input=denet, name=str(ngf * 8)+"_1")

            denet = Deconv(batch_input=denet, output_channel=ngf * 8, stride=2, name=str(ngf * 8) + "_2")
            denet = Batchnorm(batch_input=denet, name=str(ngf * 8) + "_2")

            denet = Deconv(batch_input=denet, output_channel=ngf * 8, stride=2, name=str(ngf * 8) + "_3")
            denet = Batchnorm(batch_input=denet, name=str(ngf * 8) + "_3")

            denet = Deconv(batch_input=denet, output_channel=ngf * 8, stride=2, name=str(ngf * 8) + "_4")
            denet = Batchnorm(batch_input=denet, name=str(ngf * 8) + "_4")

            denet = Deconv(batch_input=denet, output_channel=ngf * 4, stride=2, name=ngf * 4)
            denet = Batchnorm(batch_input=denet, name=ngf * 4)

            denet = Deconv(batch_input=denet, output_channel=ngf * 2, stride=2, name=ngf * 2)
            denet = Batchnorm(batch_input=denet, name=ngf * 2)

            denet = Deconv(batch_input=denet, output_channel=ngf, stride=2, name=ngf)
            denet = Batchnorm(batch_input=denet, name=ngf)

            denet = Conv(batch_input=denet, output_channel=2, stride=1, pad="SAME", name=2)
            denet = tf.nn.tanh(denet)
    return denet

def discriminator(color_img, gray_img, ndf=64):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        input_ = tf.concat([color_img, gray_img], axis=3)
        net = Conv(batch_input=input_, output_channel=ndf, stride=2, pad="SAME", name=ndf)
        net = lrelu(x=net, a=0.2)

        net = Conv(batch_input=net, output_channel=ndf * 2, stride=2, pad="SAME", name=ndf * 2)
        net = lrelu(x=net, a=0.2)

        net = Conv(batch_input=net, output_channel=ndf * 4, stride=2, pad="SAME", name=ndf * 4)
        net = lrelu(x=net, a=0.2)

        net = Conv(batch_input=net, output_channel=ndf * 8, stride=1, pad="VALID", name=ndf * 8)
        net = lrelu(x=net, a=0.2)

        net = Conv(batch_input=net, output_channel=1, stride=1, pad="VALID", name=1)
        net = tf.nn.sigmoid(net)
    return net
