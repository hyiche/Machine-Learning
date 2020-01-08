from test_con import *
import tensorflow as tf
import csv
import os
import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt

'''prepare data'''
# load tfrecord

filename_queue = tf.train.string_input_producer([path_tfrecords_test], num_epochs=test_epoch)
# read images from tfrecord
[L_image, AB_image, Mos_gray64, Mos_ab64] = read_and_decode(filename_queue)

Mos_gray_list = [Mos_gray64]
Mos_ab_list = [Mos_ab64]
Model = model.cGAN(L_image, AB_image, Mos_gray_list, Mos_ab_list)


'''check files'''
for i in [path_checkpoint, path_image_train]:
    if not os.path.exists(i):
        os.makedirs(i)


'''training model'''
# train generator
G_model = Model.train_gen(number=1, mode='test')

'''image process'''
# L-channel: [-1, 1] (float32) --> [0, 100] (float32)
# AB-channel: [-1, 1] (float32) --> [-128, 127] (float32)
grayscale_img255 = (L_image + 1)/2*255
L_img = (L_image + 1)*50
L_img_mos = (Mos_gray64 + 1)*50
targets = tf.concat([L_img_mos, Mos_ab64*128], axis=3)
AB_out = G_model.outputs
Color_out = tf.concat([L_img_mos, AB_out*128], axis=3)
Ori_tar = tf.concat([L_img, AB_image*128], axis=3)



'''start training'''
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.train.MonitoredTrainingSession(checkpoint_dir=path_checkpoint+"{}_{}/".format(name, num_epoch),
                                       save_checkpoint_steps=save_check_epoch, config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(path_checkpoint+"{}_{}/".format(name, num_epoch))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load checkpoint")

    for i in range(test_epoch):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0

        while not sess.should_stop():
            step += 1
            print(step)

            [gray_img, true_color_img, AB_gen_img, color_gen_img, ori_img] = sess.run([grayscale_img255, targets, AB_out, Color_out, Ori_tar])

            img = np.array(gray_img[0, :, :, 0], np.float64)
            img_save = img.astype(np.uint8)
            io.imsave(path_image_train + "/{}gray.png".format(step), img_save)

            img = np.array(true_color_img[0, :, :, :], np.float64)
            img = color.lab2rgb(img) * 255
            img_save = img.astype(np.uint8)
            io.imsave(path_image_train + "/{}target.png".format(step), img_save)

            img = np.array(color_gen_img[0, :, :, :], np.float64)
            img = color.lab2rgb(img) * 255
            img_save = img.astype(np.uint8)
            io.imsave(path_image_train + "/{}output.png".format(step), img_save)

            img = np.array(ori_img[0, :, :, :], np.float64)
            img = color.lab2rgb(img) * 255
            img_save = img.astype(np.uint8)
            io.imsave(path_image_train + "/{}ori.png".format(step), img_save)

            if step == max_steps:
                break

        coord.request_stop()
        coord.join(threads)
