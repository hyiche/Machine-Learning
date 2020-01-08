from configure import *
import tensorflow as tf
import numpy as np
import csv
import os
from skimage import color, io
import matplotlib.pyplot as plt

'''prepare data'''
# load tfrecord
filename_queue = tf.train.string_input_producer([path_tfrecords_train], num_epochs=data_epoch)
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
G_model = Model.train_gen(number=1, mode='train')
# train discriminator
Dreal_model = Model.train_disReal(number=1)
Dfake_model = Model.train_disFake(number=1)


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
with tf.train.MonitoredTrainingSession(checkpoint_dir=path_checkpoint+"{}_{}/".format(name, chech_epoch),
                                       save_checkpoint_steps=save_check_step, config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(path_checkpoint+"{}_{}/".format(name, chech_epoch))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load checkpoint")

    g_loss = []
    dreal_loss = []
    dfake_loss = []

    writer_g = csv.writer(open("./generator.csv", "w"))
    writer_d = csv.writer(open("./discriminator.csv", "w"))

    for i in range(num_epoch):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        loss_avg_g = 0
        loss_avg_l1 = 0
        loss_avg_dreal = 0
        loss_avg_dfake = 0
        save_loss_g = []
        save_loss_d = []

        while not sess.should_stop():
            model_g = sess.run(G_model)
            model_dreal, model_dfake = sess.run([Dreal_model, Dfake_model])

            loss_avg_g += model_g.loss
            loss_avg_l1 += model_g.l1
            loss_avg_dreal += model_dreal.loss
            loss_avg_dfake += model_dfake.loss

            step += 1

            if step == max_steps:
                loss_avg_g /= max_steps
                loss_avg_l1 /= max_steps
                loss_avg_dreal /= max_steps
                loss_avg_dfake /= max_steps

                save_loss_g.append(loss_avg_g)
                save_loss_g.append(loss_avg_l1)
                save_loss_d.append(loss_avg_dreal)
                save_loss_d.append(loss_avg_dfake)

                g_loss.append(loss_avg_g)
                dreal_loss.append(loss_avg_dreal)
                dfake_loss.append(loss_avg_dfake)

                writer_g.writerow(save_loss_g)
                writer_d.writerow(save_loss_d)

                if i % print_epoch == 0 or i == num_epoch-1:
                    print("epoch:{}, gloss:{}, l1:{}, dloss_real:{}, dloss_fake:{}".format(i, loss_avg_g, loss_avg_l1, loss_avg_dreal, loss_avg_dfake))

                if i % save_img_epoch == 0 or i == num_epoch-1:
                    [gray_img, true_color_img, AB_gen_img, color_gen_img, ori_img] = sess.run([grayscale_img255, targets, AB_out, Color_out, Ori_tar])

                    img = np.array(gray_img[0, :, :, 0], np.float64)
                    img_save = img.astype(np.uint8)
                    io.imsave(path_image_train + "/{}gray.png".format(i), img_save)

                    img = np.array(true_color_img[0, :, :, :], np.float64)
                    img = color.lab2rgb(img) * 255
                    img_save = img.astype(np.uint8)
                    io.imsave(path_image_train + "/{}target.png".format(i), img_save)

                    img = np.array(color_gen_img[0, :, :, :], np.float64)
                    img = color.lab2rgb(img) * 255
                    img_save = img.astype(np.uint8)
                    io.imsave(path_image_train + "/{}output.png".format(i), img_save)

                    img = np.array(ori_img[0, :, :, :], np.float64)
                    img = color.lab2rgb(img) * 255
                    img_save = img.astype(np.uint8)
                    io.imsave(path_image_train + "/{}ori.png".format(i), img_save)

                break

        coord.request_stop()
        coord.join(threads)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(g_loss)
fig.savefig('./gen.png')
plt.close(fig)

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(dreal_loss)
fig1.savefig('./d_real.png')
plt.close(fig1)

fig2, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.plot(dfake_loss)
fig2.savefig('./d_fake.png')
plt.close(fig2)
