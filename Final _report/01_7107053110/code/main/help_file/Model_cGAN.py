from .create_model import *
import collections
import numpy as np
import tensorflow as tf

name = "Markov_6AE_sep_GAN_1"

EPS = 1e-12
# initial learning rate for adam
lr = 1e-4
# momentum term of adam
beta_1 = 0.5
# lambda
lam = 100

Model_gen = collections.namedtuple("Model_gen1", "loss, l1, train, step, outputs")
Model_disReal = collections.namedtuple("Model_disReal", "loss, train, step")
Model_disFake = collections.namedtuple("Model_disFake", "loss, train, step")


class cGAN(object):
    def __init__(self, L_image, AB_image, Mos_gray_list, Mos_color_list):
        self.L_img = L_image
        self.AB_img = AB_image
        self.Mos_gray = Mos_gray_list
        self.Mos_color = Mos_color_list
        self.step_g_dic = {"1": tf.train.get_or_create_global_step(), "2": tf.train.get_or_create_global_step(),
                           "3": tf.train.get_or_create_global_step(), "4": tf.train.get_or_create_global_step(),
                           "5": tf.train.get_or_create_global_step(), "6": tf.train.get_or_create_global_step()}
        self.step_dReal = tf.train.get_or_create_global_step()
        self.step_dFake = tf.train.get_or_create_global_step()

    # def create_gen_dis1(self):
    #     # generator
    #     self.AB_gen1 = generator(L_img=self.L_img, name=1, Mos_gray_img=self.Mos_gray[0])
    #     # discriminator real
    #     self.predictReal1 = discriminator(color_img=self.Mos_color[0], gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake1 = discriminator(color_img=self.AB_gen1, gray_img=self.L_img)
    #
    # def create_gen_dis2(self):
    #     # generator
    #     self.AB_gen2 = generator(L_img=self.L_img, name=2, Mos_gray_img=self.Mos_gray[1], Mos_gen_img=self.Mos_color[0])
    #     # discriminator real
    #     self.predictReal2 = discriminator(color_img=self.Mos_color[1], gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake2 = discriminator(color_img=self.AB_gen2, gray_img=self.L_img)
    #
    # def create_gen_dis3(self):
    #     # generator
    #     self.AB_gen3 = generator(L_img=self.L_img, name=3, Mos_gray_img=self.Mos_gray[2], Mos_gen_img=self.Mos_color[1])
    #     # discriminator real
    #     self.predictReal3 = discriminator(color_img=self.Mos_color[2], gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake3 = discriminator(color_img=self.AB_gen3, gray_img=self.L_img)
    #
    # def create_gen_dis4(self):
    #     # generator
    #     self.AB_gen4 = generator(L_img=self.L_img, name=4, Mos_gray_img=self.Mos_gray[3], Mos_gen_img=self.Mos_color[2])
    #     # discriminator real
    #     self.predictReal4 = discriminator(color_img=self.Mos_color[3], gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake4 = discriminator(color_img=self.AB_gen4, gray_img=self.L_img)
    #
    # def create_gen_dis5(self):
    #     # generator
    #     self.AB_gen5 = generator(L_img=self.L_img, name=5, Mos_gray_img=self.Mos_gray[4], Mos_gen_img=self.Mos_color[3])
    #     # discriminator real
    #     self.predictReal5 = discriminator(color_img=self.Mos_color[4], gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake5 = discriminator(color_img=self.AB_gen5, gray_img=self.L_img)
    #
    # def create_gen_dis6(self):
    #     # generator
    #     self.AB_gen6 = generator(L_img=self.L_img, name=6, Mos_gen_img=self.Mos_color[4])
    #     # discriminator real
    #     self.predictReal6 = discriminator(color_img=self.AB_img, gray_img=self.L_img)
    #     # discriminator fake
    #     self.predictFake6 = discriminator(color_img=self.AB_gen6, gray_img=self.L_img)

    def create_gen_dis(self, number, prev_mos_gen=None):
        if number == 1:
            mos_gray = self.Mos_gray[0]
            mos_color = self.Mos_color[0]
        elif number == 6:
            mos_gray = None
            mos_color = self.AB_img
        else:
            mos_gray = self.Mos_gray[number - 1]
            mos_color = self.Mos_color[number - 1]

        # generator
        AB_gen = generator(L_img=self.L_img, name=number, Mos_gen_img=prev_mos_gen, Mos_gray_img=mos_gray)
        # discriminator real
        predictReal = discriminator(color_img=mos_color, gray_img=self.L_img)
        # discriminator fake
        predictFake = discriminator(color_img=AB_gen, gray_img=self.L_img)

        return [AB_gen, predictReal, predictFake]


    def train_gen(self, number, mode="train"):
        AB_ori = self.Mos_color[number - 1] if number < 6 else self.AB_img
        [AB_gen, _, predictFake] = self.create_gen_dis(number=number)
        gen_tvar = [var for var in tf.trainable_variables() if "generator_{}".format(str(number)) in var.name]
        gen_loss_GAN = tf.reduce_mean(-tf.log(predictFake + EPS))
        # gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictFake, labels=tf.ones_like(predictFake)))
        gen_loss_L1 = tf.reduce_mean(tf.abs(AB_ori - AB_gen))
        gen_loss_total = gen_loss_GAN + lam * gen_loss_L1
        gen_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1, name="op_g_{}".format(str(number)))
        gen_train = gen_optim.minimize(loss=gen_loss_total, var_list=gen_tvar,
                                       global_step=self.step_g_dic["{}".format(str(number))],
                                       name="train_g_{}".format(str(number)))
        update_gen_op = tf.assign_add(self.step_g_dic["{}".format(str(number))], 1)

        if mode == "train":
            return Model_gen(loss=gen_loss_total, l1=gen_loss_L1, train=gen_train, step=update_gen_op, outputs=AB_gen)
        elif mode == "test":
            return Model_gen(loss=None, l1=None, train=None, step=None, outputs=AB_gen)

    def train_disReal(self, number):
        [_, predictReal, __] = self.create_gen_dis(number=number)
        dis_tvars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        dis_lossReal = tf.reduce_mean(-tf.log(predictReal + EPS))
        # dis_lossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictReal, labels=tf.ones_like(predictReal)))
        dis_optimReal = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1, name="op_real")
        dis_trainReal = dis_optimReal.minimize(loss=dis_lossReal, var_list=dis_tvars, global_step=self.step_dReal,
                                               name="train_real")
        update_real_op = tf.assign_add(self.step_dReal, 1)

        return Model_disReal(loss=dis_lossReal,
                             train=dis_trainReal,
                             step=update_real_op)

    def train_disFake(self, number):
        [_, __, predictFake] = self.create_gen_dis(number=number)
        dis_tvars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        dis_lossFake = tf.reduce_mean(-tf.log(1 - predictFake + EPS))
        # dis_lossFake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictFake, labels=tf.zeros_like(predictFake)))
        dis_optimFake = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1, name="op_fake")
        dis_trainFake = dis_optimFake.minimize(loss=dis_lossFake, var_list=dis_tvars, global_step=self.step_dFake,
                                               name="train_fake")
        update_fake_op = tf.assign_add(self.step_dFake, 1)

        return Model_disReal(loss=dis_lossFake,
                             train=dis_trainFake,
                             step=update_fake_op)
