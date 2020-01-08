import yfinance as yf
from pandas_datareader import data as pdr
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import tensorflow.contrib as cb
import datetime as dt
#component test 在882
tf.disable_eager_execution()
plt.rcParams['axes.unicode_minus'] = False
def append_loss_to_txt_each_epoch(name, input):
    a = open(name+'.txt','a')
    a.write(str(np.mean(input.copy()))+" ")
    a.close()
def parse_args():
    parser = argparse.ArgumentParser("Hyper-parameters for training model implementation")
    # Core training parameters
    parser.add_argument("--lr_act", type=float, default=0.00000001, help="learning rate for Q-action")
    parser.add_argument("--lr_num", type=float, default=0.0001, help="learning rate for R-num")
    parser.add_argument("--lr_e2e", type=float, default=0.00000001, help="learning rate for e2e")
    parser.add_argument("--gamma", type=float, default=0.5, help="discount factor for td-error of both Q and R")
    parser.add_argument("--batch_size", type=int, default=4, help="number of batches")
    parser.add_argument("--pre1_epochs", type=int, default=3, help="pre1")
    parser.add_argument("--pre2_epochs", type=int, default=3, help="pre2")
    parser.add_argument("--main_epochs", type=int, default=3, help="main")
    parser.add_argument("--test_epochs", type=int, default=1, help="test")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--n_features", type=int, default=200, help="number of features")
    parser.add_argument("--idx", type=int, default=0.6, help="split %")
    return parser.parse_args(args=[])
def parse_args2():
    parser2 = argparse.ArgumentParser("Hyper-parameters for training model implementation")
    # Core training parameters
    parser2.add_argument("--batch_size", type=int, default=4, help="number of batches")
    parser2.add_argument("--n_epochs", type=int, default=1, help="number of epochs")
    parser2.add_argument("--n_features", type=int, default=200, help="number of features")
    parser2.add_argument("--pre1_epochs", type=int, default=1, help="pre1")
    parser2.add_argument("--pre2_epochs", type=int, default=1, help="pre2")
    parser2.add_argument("--main_epochs", type=int, default=1, help="main")
    parser2.add_argument("--test_epochs", type=int, default=1, help="test")
    parser2.add_argument("--idx", type=int, default=0.8, help="split %")
    parser2.add_argument("--max_profit_act", type=int, default=0, help="max_profit_act")
    parser2.add_argument("--max_profit_pos", type=int, default=0, help="max_profit_pos")
    return parser2.parse_args(args=[])
class Model_act(object):
    def __init__(self, arglist,sess):
        self.sess=sess
        self.gamma = arglist.gamma
        self.lr_act = arglist.lr_act
        self.lr_num = arglist.lr_num
        self.lr_e2e = arglist.lr_e2e
        self.n_features = arglist.n_features
        self.batch_size = arglist.batch_size
        with g1.as_default():
            self.i = tf.placeholder(dtype=tf.int32, name="i")
            self.s = tf.placeholder(dtype=tf.float64, shape=[None, self.n_features-2], name="input")
            self.s_ = tf.placeholder(dtype=tf.float64, shape=[None, self.n_features - 2], name="input_")
            self.r = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="reward")
            self.q_ = tf.placeholder(dtype=tf.float64, shape=[None,1], name="q_next")
            self.ones = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="ones")
            self.a_tm1_act = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="a_tm1")
            self.p_t = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="p_t")
            self.p_tm1 = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="p_tm1")
            self.p_tm200 = tf.placeholder(dtype=tf.float64, shape=[None, ], name="p_tm200")
            self.per_tm1 = tf.placeholder(dtype=tf.float64, shape=[None,1 ], name="per_tm1")
            self.pred_tm1 = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="pred_tm1")
            with tf.variable_scope('common_network_p', reuse=None):
                fc1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu, name='fc1',kernel_regularizer=cb.layers.l2_regularizer(0.001))
            with tf.variable_scope('act_network_p', reuse=None):
                act_fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu, name='act_fc2')
                act_fc3 = tf.layers.dense(inputs=act_fc2, units=20, activation=tf.nn.relu, name='act_fc3')
                self.act_out = tf.layers.dense(inputs=act_fc3, units=3, activation=None, name='act_out')
                self.action = tf.math.argmax(input=self.act_out, axis=1)#[0]
                self.max_act_q = tf.reduce_max(self.act_out,axis=1)#argmax
            with tf.variable_scope('common_network_p', reuse=tf.AUTO_REUSE):
                fc1_ = tf.layers.dense(inputs=self.s_, units=100, activation=tf.nn.relu, name='fc1',kernel_regularizer=cb.layers.l2_regularizer(0.001))
            with tf.variable_scope('act_network_p', reuse=tf.AUTO_REUSE):
                act_fc2_ = tf.layers.dense(inputs=fc1_, units=50, activation=tf.nn.relu, name='act_fc2')
                act_fc3_ = tf.layers.dense(inputs=act_fc2_, units=20, activation=tf.nn.relu, name='act_fc3')
                self.act_out_ = tf.layers.dense(inputs=act_fc3_, units=3, activation=None, name='act_out')
                self.action_ = tf.math.argmax(input=self.act_out_, axis=1)  # [0]
                self.max_act_q_= tf.reduce_max(self.act_out_, axis=1)  # argmax
                self.act_td_error = self.r + self.gamma * self.max_act_q_ - self.max_act_q
                self.act_loss = tf.math.reduce_sum(tf.math.square(self.act_td_error))
            with tf.variable_scope('num_network_p', reuse=None):
                num_fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu, name='num_fc2',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                num_fc3 = tf.layers.dense(inputs=num_fc2, units=20, activation=tf.nn.sigmoid, name='num_fc3',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.per = tf.layers.dense(inputs=num_fc3, units=1, activation=None, name='num_out',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.merge = tf.concat([tf.ones([64, ], dtype=tf.float64) +self.per_tm1 * 10, tf.ones([64, ], dtype=tf.float64) +self.pred_tm1 * 10], 1)
                self.num_out = ( tf.reduce_min(self.merge, axis=1)) / (
                         tf.reduce_max(self.merge, axis=1))
                self.num_q = self.num_out
                # loss_reg = tf.losses.get_regularization_losses()
                # loss_reg = tf.math.reduce_sum(loss_reg)
                # loss_reg = tf.constant(loss_reg, dtype=tf.float64)
                self.num_loss = tf.math.reduce_sum(
                    tf.math.square((self.per - (self.p_t - self.p_tm1) / self.p_tm1) * 20))
            with tf.variable_scope('train_p', reuse=None):
                only_act_var = tf.trainable_variables(scope='common_network_p') + tf.trainable_variables(scope='act_network_p')
                only_num_var = tf.trainable_variables(scope='num_network_p')
                self.act_train_op = tf.train.AdamOptimizer(self.lr_act).minimize(self.act_loss, var_list=only_act_var)
                self.num_train_op = tf.train.AdamOptimizer(self.lr_num).minimize(self.num_loss, var_list=only_num_var)
                self.all_act_train_op = tf.train.AdamOptimizer(self.lr_e2e).minimize(self.act_loss)
                self.all_num_train_op = tf.train.AdamOptimizer(self.lr_e2e).minimize(self.num_loss)
            self.saver = tf.train.Saver(max_to_keep=30)
            # self.sess.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess, "1_index_^GSPC_sess1.ckpt")
        self.saver.restore(self.sess, "0103"+"_sess2.ckpt")
    def get_per(self, s,s_, p_t, p_tm1, per_tm1, pred_tm1):
        per = self.sess.run([self.per], feed_dict={self.s: s,self.s_:s_, self.p_t: p_t, self.p_tm1: p_tm1, self.per_tm1: per_tm1,
                                                    self.pred_tm1: pred_tm1})
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        per = np.asarray(per)
        per = per.reshape((arglist.batch_size, 1))
        return per
    def get_qacts_and_num(self, s,s_,p_t,p_tm1,per_tm1,pred_tm1):
        q_acts, num = self.sess.run([self.act_out, self.num_q], feed_dict={self.s: s,self.s_:s_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1})
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        return q_acts, num

    # 把q轉動作跟數量
    def get_q_score(self, s,s_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("s.shape: ", s.shape)
        max_act_q, num_q = self.sess.run([self.max_act_q, self.num_q], feed_dict={self.s: s,self.s_:s_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1})
        return max_act_q, num_q

    def pretrain_1st(self, r, s,s_, q_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.r: r, self.s: s, self.s_:s_,self.q_: q_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, act_loss = self.sess.run([self.act_train_op, self.act_loss], feed_dict=feed_dict)
        return act_loss

    def pretrain_2nd(self,s,s_, ones, a_tm1_act, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.s:s,self.s_:s_,self.ones:ones , self.a_tm1_act:a_tm1_act, self.p_t:p_t, self.p_tm1:p_tm1, self.p_tm200:p_tm200,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, num_loss = self.sess.run([self.num_train_op, self.num_loss], feed_dict=feed_dict)
        return num_loss

    def act_update(self, r, s,s_, q_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.r: r, self.s: s,self.s_:s_, self.q_: q_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, act_loss = self.sess.run([self.all_act_train_op, self.act_loss], feed_dict=feed_dict)
        return act_loss

    def num_update(self,s,s_, ones, a_tm1_act, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.s:s,self.s_:s_,self.ones:ones , self.a_tm1_act:a_tm1_act, self.p_t:p_t, self.p_tm1:p_tm1, self.p_tm200:p_tm200,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, num_loss = self.sess.run([self.all_num_train_op, self.num_loss], feed_dict=feed_dict)
        return num_loss

    def main_train(self, r, s, s_,act_q_, ones, a_tm1_act, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        act_loss = self.act_update(r=r, s=s,s_=s_, q_=act_q_, p_t=p_t, p_tm1=p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1)
        num_loss = self.num_update(s=s,s_=s_,ones=ones , a_tm1_act=a_tm1_act, p_t=p_t, p_tm1=p_tm1, p_tm200=p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1)
        return act_loss, num_loss
class Model_pos(object):
    def __init__(self, arglist, sess2):
        self.sess2=sess2
        self.gamma = arglist.gamma
        self.lr_act = arglist.lr_act
        self.lr_num = arglist.lr_num
        self.lr_e2e = arglist.lr_e2e
        self.n_features = arglist.n_features
        self.batch_size = arglist.batch_size

        with g2.as_default():
            self.i=tf.placeholder(dtype=tf.int32, name="i")
            self.s = tf.placeholder(dtype=tf.float64, shape=[None, self.n_features-2], name="input")
            self.s_ = tf.placeholder(dtype=tf.float64, shape=[None, self.n_features - 2], name="input_")
            self.r = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="reward")
            self.q_ = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="q_next")
            self.a_tm1_pos = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="a_tm1_pos")
            self.a_tm2_pos = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="a_tm2_pos")
            self.num_tm2_pos = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="num_tm2_pos")
            self.pos_tm2_pos = tf.placeholder(dtype=tf.float64, shape=[None,1 ], name="pos_tm2_pos")
            self.ones= tf.placeholder(dtype=tf.float64, shape=[None, 1], name="ones")
            self.p_t = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="p_t")
            self.p_tm1 = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="p_tm1")
            self.p_tm200 = tf.placeholder(dtype=tf.float64, shape=[None, ], name="p_tm200")
            self.per_tm1 = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="per_tm1")
            self.pred_tm1 = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="pred_tm1")
            with tf.variable_scope('common_network_p', reuse=None):
                fc1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu, name='fc1',kernel_regularizer=cb.layers.l2_regularizer(0.001))
            with tf.variable_scope('act_network_p', reuse=None):
                act_fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu, name='act_fc2')
                act_fc3 = tf.layers.dense(inputs=act_fc2, units=20, activation=tf.nn.relu, name='act_fc3')
                self.act_out = tf.layers.dense(inputs=act_fc3, units=3, activation=None, name='act_out')
                self.action = tf.math.argmax(input=self.act_out, axis=1)  # [0]
                self.max_act_q = tf.reduce_max(self.act_out, axis=1)  # argmax
            with tf.variable_scope('common_network_p', reuse=tf.AUTO_REUSE):
                fc1_ = tf.layers.dense(inputs=self.s_, units=100, activation=tf.nn.relu, name='fc1',kernel_regularizer=cb.layers.l2_regularizer(0.001))
            with tf.variable_scope('act_network_p', reuse=tf.AUTO_REUSE):
                act_fc2_ = tf.layers.dense(inputs=fc1_, units=50, activation=tf.nn.relu, name='act_fc2',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                act_fc3_ = tf.layers.dense(inputs=act_fc2_, units=20, activation=tf.nn.relu, name='act_fc3',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.act_out_ = tf.layers.dense(inputs=act_fc3_, units=3, activation=None, name='act_out',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.action_ = tf.math.argmax(input=self.act_out_, axis=1)  # [0]
                self.max_act_q_ = tf.reduce_max(self.act_out_, axis=1)  # argmax
                self.act_td_error = self.r + self.gamma * self.max_act_q_ - self.max_act_q
                self.act_loss = tf.math.reduce_sum(tf.math.square(self.act_td_error))
            with tf.variable_scope('num_network_p', reuse=None):
                num_fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu, name='num_fc2',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                num_fc3 = tf.layers.dense(inputs=num_fc2, units=20, activation=tf.nn.sigmoid, name='num_fc3',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.per = tf.layers.dense(inputs=num_fc3, units=1, activation=None, name='num_out',kernel_regularizer=cb.layers.l2_regularizer(0.001))
                self.merge = tf.concat([tf.ones([64, ], dtype=tf.float64) + self.per_tm1 * 10,
                                        tf.ones([64, ], dtype=tf.float64) + self.pred_tm1 * 10], 1)
                self.num_out = (tf.reduce_min(self.merge, axis=1)) / (
                    tf.reduce_max(self.merge, axis=1))
                self.num_q = self.num_out
                # loss_reg=tf.losses.get_regularization_losses()
                # loss_reg=tf.math.reduce_sum(loss_reg)
                # loss_reg=tf.constant(loss_reg,dtype=tf.float64)
                self.num_loss = tf.math.reduce_sum(
                    tf.math.square((self.per - (self.p_t - self.p_tm1) / self.p_tm1)*20))
                # self.output_list=[]
                # def f1(pos_tm2_pos, num_out):
                #     result = tf.cond(tf.abs(pos_tm2_pos) <= num_out, lambda: tf.constant(0, dtype=tf.float64),
                #                      lambda: tf.sign(pos_tm2_pos) * tf.abs(pos_tm2_pos - num_out))
                #     return result
                # i = tf.constant(0)
                # while_condition = lambda i: tf.less(i, 64)
                # def body(i):
                #     a=(tf.cond(tf.cast(self.num_out[i][0] != tf.constant(0,dtype=tf.float64), tf.bool),
                #                            lambda: (self.a_tm1_pos[i][0] * self.num_out[i][0]),
                #                            lambda: (f1(self.pos_tm2_pos[i][0], self.num_out[i][0]))))
                #     self.output_list.append(a)
                #     return [tf.add(i, 1)]
                # # do the loop:
                # r = tf.while_loop(while_condition, body, [i])
                # outputs = tf.stack(self.output_list)


                #####self.num_loss = -tf.math.reduce_sum((self.ones + self.a_tm1_pos*self.num_out*10 * (self.p_t - self.p_tm1) / self.p_tm1) * self.p_tm1 / self.p_tm200)
                # pos_pos=self.pos_pos(self.s, self.a_tm1_pos, self.num_out*10, self.a_tm2_pos, self.num_tm2_pos, self.pos_tm2_pos)
                # self.num_loss = -tf.math.reduce_sum((self.ones + pos_pos * (self.p_t - self.p_tm1) / self.p_tm1) * self.p_tm1 / self.p_tm200)

            with tf.variable_scope('train_p', reuse=None):
                only_act_var = tf.trainable_variables(scope='common_network_p') + tf.trainable_variables(scope='act_network_p')
                only_num_var = tf.trainable_variables(scope='num_network_p')
                self.act_train_op = tf.train.AdamOptimizer(self.lr_act).minimize(self.act_loss, var_list=only_act_var)
                self.num_train_op = tf.train.AdamOptimizer(self.lr_num).minimize(self.num_loss, var_list=only_num_var)
                self.all_act_train_op = tf.train.AdamOptimizer(self.lr_e2e).minimize(self.act_loss)
                self.all_num_train_op = tf.train.AdamOptimizer(self.lr_e2e).minimize(self.num_loss)
            self.saver = tf.train.Saver(max_to_keep=30)
            # self.sess2.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess2, "1_index_^GSPC_sess2.ckpt")
        self.saver.restore(self.sess2, "0103"+"_sess2.ckpt")

    # def pos_pos(self,s,a_tm1_pos, a_tm2_pos, num_tm2_pos, pos_tm2_pos):
    #     num_tm1_pos=self.sess2.run([self.num_out], feed_dict={self.s: s})
    #     position = np.zeros((arglist.batch_size, 1))
    #     what_position = np.zeros((arglist.batch_size, 1))
    #     num = np.zeros((arglist.batch_size, 1))
    #     for i in range(len(position)):
    #         if a_tm1_pos[i] == 0:
    #             what_position[i] = f(pos_tm2_pos[i])
    #             if a_tm2_pos[i] == 0:
    #                 num[i] = num_tm1_pos[i] - num_tm2_pos[i]  # 要平倉的數
    #                 if abs(pos_tm2_pos[i]) < num[i]:
    #                     position[i] = 0
    #                 if num[i] <= 0:
    #                     position[i] = pos_tm2_pos[i]
    #                 else:
    #                     position[i] = what_position[i] * (abs(pos_tm2_pos[i]) - num[i])
    #             else:
    #                 if abs(pos_tm2_pos[i]) < num_tm1_pos[i]:
    #                     position[i] = 0
    #                 else:
    #                     position[i] = what_position[i] * (abs(pos_tm2_pos[i]) - num_tm1_pos[i])
    #         else:
    #             position[i] = a_tm1_pos[i] * num_tm1_pos[i]
    #     # print("pos:", position, sep="")
    #     return position
    def get_per(self, s,s_,p_t,p_tm1,per_tm1,pred_tm1):
        per = self.sess2.run([self.per], feed_dict={self.s: s,self.s_: s_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1})
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        per = np.asarray(per)
        per = per.reshape((arglist.batch_size,1))
        return per

    def get_qacts_and_num(self, s ,s_,p_t,p_tm1,per_tm1,pred_tm1):
        q_acts, num = self.sess2.run([self.act_out, self.num_q], feed_dict={self.s: s,self.s_: s_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1})
        # print('q_acts', q_acts)
        return q_acts, num

    # 把q轉動作跟數量
    def get_q_score(self, s,s_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("s.shape: ", s.shape)
        max_act_q, num_q = self.sess2.run([self.max_act_q, self.num_q], feed_dict={self.s: s,self.s_: s_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1})
        return max_act_q, num_q

    def pretrain_1st(self, r, s,s_, q_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.r: r, self.s: s,self.s_: s_, self.q_: q_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, act_loss = self.sess2.run([self.act_train_op, self.act_loss], feed_dict=feed_dict)
        return act_loss

    def pretrain_2nd(self, s,s_,a_tm1_pos,  ones, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.s:s,self.s_: s_,self.a_tm1_pos: a_tm1_pos,  self.ones:ones, self.p_t:p_t, self.p_tm1:p_tm1, self.p_tm200:p_tm200,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, num_loss = self.sess2.run([self.num_train_op, self.num_loss], feed_dict=feed_dict)
        return num_loss

    def act_update(self, r, s,s_, q_,p_t,p_tm1,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.r: r, self.s: s,self.s_: s_, self.q_: q_,self.p_t:p_t,self.p_tm1:p_tm1,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, act_loss = self.sess2.run([self.all_act_train_op, self.act_loss], feed_dict=feed_dict)
        return act_loss

    def num_update(self,s,s_, a_tm1_pos,  ones, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        # print("r.shape: ", r.shape, "s.shape: ", s.shape, "q_.shape: ", q_.shape)
        feed_dict = {self.s:s,self.s_: s_, self.a_tm1_pos: a_tm1_pos,  self.ones:ones, self.p_t:p_t, self.p_tm1:p_tm1, self.p_tm200:p_tm200,self.per_tm1:per_tm1,self.pred_tm1:pred_tm1}
        _, num_loss = self.sess2.run([self.all_num_train_op, self.num_loss], feed_dict=feed_dict)
        return num_loss

    def main_train(self, r, s,s_, act_q_, a_tm1_pos,  ones, p_t, p_tm1, p_tm200,per_tm1,pred_tm1):
        act_loss = self.act_update(r=r, s=s,s_=s_, q_=act_q_, p_t=p_t, p_tm1=p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1)
        num_loss = self.num_update(s=s,s_=s_, a_tm1_pos=a_tm1_pos,  ones=ones, p_t=p_t, p_tm1=p_tm1, p_tm200=p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1)
        return act_loss, num_loss
def subplt(r,c,place,l,x,y,name):
    plt.subplot(r,c,place)
    plt.plot(l)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(name,y=1.05)
def plt_save(act, pos, xl, yl, name):
    plt.figure()
    a=name
    plt.title(a,y=1.05)
    plt.yscale("linear")
    # plt.ylim(-5000,5000)
    plt.xlabel(xl)
    plt.ylabel(yl)
    l1, = plt.plot(act, color='blue',linewidth=0.5)
    l2, = plt.plot(pos, color='red',linewidth=0.5)
    plt.legend(handles=[l1, l2], labels=['act', 'pos'], loc='best')
    plt.savefig(a)
    plt.close()
    # plt.show()
def plt_loss(train,test, name):
    plt.figure()
    a=name
    plt.title(a,y=1.05)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    l1, = plt.plot(train, color='blue', linewidth=0.5)
    l2, = plt.plot(test, color='red', linewidth=0.5)
    plt.legend(handles=[l1, l2], labels=['train', 'test'], loc='best')
    plt.savefig(a)
    plt.close()
    # plt.show()
def plt_num_act(act, num, xl, yl, name):
    plt.figure()
    plt.title(name,y=1.05)
    plt.xlabel(xl)
    plt.ylabel(yl)
    legend=["r","b","g"]
    label=["+1","0","-1"]
    recs=[]
    for i in range(len(legend)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=legend[i]))
    colors=list()
    for i in range(len(act)):
        if act[i]==-1:
            colors.append("g")
        elif act[i]==0:
            colors.append("b")
        else:
            colors.append("r")
    plt.scatter(x=np.asarray([i for i in range(len(act))]),y=act,s=00.1,color=colors)
    plt.scatter(x=np.asarray([i for i in range(len(act))]), y=num, s=00.1, color=colors)
    plt.legend(recs,label)
    plt.savefig(name)
    plt.close()
    # plt.show()
def plt_total_profit(p,act,pos, name):
    plt.figure()
    a=name
    plt.title(a,y=1.05)
    plt.xlabel("time")
    plt.ylabel("accumulated investment profit(%)")
    l1, = plt.plot(p, color='black', linewidth=0.5)
    l2, = plt.plot(act, color='blue',linewidth=0.5)
    l3, = plt.plot(pos, color='red',linewidth=0.5)

    plt.legend(handles=[l1, l2, l3], labels=['market','act', 'pos'], loc='best')
    plt.savefig(a)
    plt.close()
    # plt.show()
def plt_price(p,name):
    plt.figure()
    a=name
    plt.title(a,y=1.05)
    plt.xlabel("time")
    plt.ylabel("accumulated price fluctuation(%)")
    plt.plot(p, color='black',linewidth=0.5)
    plt.savefig(a)
    plt.close()
    # plt.show()
def download_data(code):
    yf.pdr_override()
    data_frame = pdr.get_data_yahoo(code,start = dt.datetime(2012, 5, 18))
    close_data = data_frame.Close
    if code=="HSI":
        close_data=close_data[close_data<150000]
        df=close_data[:3500]
        df1=close_data[3500:7000]
        df1 = df1[df1>5000]
        df2=close_data[6800:]
        df2=df2[df2>17000]
        c=pd.merge(df,df1,how='outer')
        close_data=pd.merge(c,df2,how='outer')
    per_data= close_data.copy()
    # print(per_data)
    # print(dff)
    for i in range(len(per_data)):
        if i == 0:
            continue
        else:
            per_data[i]=(close_data[i]-close_data[i-1])/close_data[i-1]
    per_data[0]=0
    ####################
    close_data = close_data.tolist()  # tolist() 轉換成list
    per_data = per_data.tolist()
    data, front_index, rear_index = [], 0, arglist.n_features - 1  # front資料左邊 rear資料右邊 #n_feature一次抓幾筆
    while rear_index <= len(close_data) - 1:
        data.append(close_data[front_index: rear_index + 1])  # 價差取200 一次棟一天
        front_index += 1
        rear_index += 1
    data = np.asarray(data)
    # per
    data2, front_index, rear_index = [], 0, arglist.n_features - 1
    while rear_index <= len(close_data) - 1:
        data2.append(per_data[front_index: rear_index + 1])  # 價差取200 一次棟一天
        front_index += 1
        rear_index += 1
    data2 = np.asarray(data2)

    per_198 = data2[:, 1:199]
    p_t, p_tm200, p_tm1 = data[:, -1], data[:, 0], data[:, -2]

    ####################
    def data_split(per_198, p_tm200, p_t, p_tm1, arglist):  # 200個一排 一次動一個 #200天前 #200天之後可以開始實驗了
        idx = int(len(per_198) * arglist.idx)
        train_per_198, train_p_t, train_p_tm200, train_p_tm1 = per_198[:idx], p_t[:idx], p_tm200[:idx], p_tm1[:idx]
        test_per_198, test_p_t, test_p_tm200, test_p_tm1 = per_198[idx:], p_t[idx:], p_tm200[idx:], p_tm1[idx:]
        return train_per_198, train_p_t, train_p_tm200, test_per_198, test_p_t, test_p_tm200, train_p_tm1, test_p_tm1

    train_per_198, train_p_t, train_p_tm200, test_per_198, test_p_t, test_p_tm200, train_p_tm1, test_p_tm1 = data_split(
        per_198, p_tm200, p_t, p_tm1, arglist=arglist)
    return train_per_198, train_p_t, train_p_tm200,train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1
def batch_train(train_per_198, train_p_t, train_p_tm200,train_p_tm1, arglist,k):

    def batch_gen(x, arglist):
        data = np.asarray(x[k:k+arglist.batch_size])
        return data
    def batch_gen_(x, arglist):
        data = np.asarray(x[k+1:k +1+ arglist.batch_size])
        return data
    train_per_198,train_per_198_, train_p_t, train_p_tm200, train_p_tm1 = batch_gen(train_per_198, arglist=arglist),batch_gen_(train_per_198, arglist=arglist), batch_gen(train_p_t, arglist=arglist), batch_gen(
        train_p_tm200, arglist=arglist), batch_gen(train_p_tm1, arglist=arglist)
    ##batch_gen

    return train_per_198,train_per_198_, train_p_t, train_p_tm200,  train_p_tm1
def batch_test(test_per_198, test_p_t, test_p_tm200, test_p_tm1, arglist,k):

    def batch_gen(x, arglist):
        data = np.asarray(x[k:k + arglist.batch_size])
        return data
    def batch_gen_(x, arglist):
        data = np.asarray(x[k+1:k +1+ arglist.batch_size])
        return data

    test_per_198,test_per_198_, test_p_t, test_p_tm200, test_p_tm1 = batch_gen(test_per_198, arglist=arglist),batch_gen_(test_per_198, arglist=arglist), batch_gen(
        test_p_t, arglist=arglist), batch_gen(
        test_p_tm200, arglist=arglist), batch_gen(test_p_tm1, arglist=arglist)
    ##batch_gen

    return test_per_198, test_per_198_, test_p_t, test_p_tm200, test_p_tm1

def get_a_and_num(q_acts, num):
    # print(q_acts[-1])
    a=np.argmax(q_acts,axis=1) - 1
    a=a[ :,np.newaxis]
    num=num
    for numi in range(len(num)):
        if num[numi]<0:
            num[numi]=0
        elif num[numi]>1:
            num[numi]=1
        else:
            num[numi]=num[numi]
    return a , num *10
def f(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
def reward_pos(p_t, p_tm1, p_tm200, num_tm1_pos, pos_tm1_pos):
    result=np.zeros((arglist.batch_size,1))
    for i in range(len(result)):
        result[i] = (1 + pos_tm1_pos[i]*(p_t[i]-p_tm1[i])/p_tm1[i])*p_tm1[i] / p_tm200[i]
    return result
def reward_act(p_t, p_tm1, p_tm200, num_tm1_act, a_tm1_act):
    p_tm200,num_tm1_act=   p_tm200[:,np.newaxis],num_tm1_act[:,np.newaxis]
    # print(p_t.shape, p_tm1.shape, p_tm200.shape, num_tm1_act.shape, a_tm1_act.shape)
    result = num_tm1_act*(np.ones((arglist.batch_size,1)) +a_tm1_act*(p_t-p_tm1)/p_tm1)*p_tm1/p_tm200
    return result
def pos_pos(a_tm1_pos, num_tm1_pos, pos_tm1_pos, a_tm2_pos, num_tm2_pos, pos_tm2_pos):
    position=np.zeros((arglist.batch_size,1))
    what_position = np.zeros((arglist.batch_size, 1))
    num=np.zeros((arglist.batch_size, 1))
    for i in range(len(position)):
        if a_tm1_pos[i] == 0:
            what_position[i] = f(pos_tm2_pos[i])
            if a_tm2_pos[i] == 0:
                num[i] = num_tm1_pos[i] - num_tm2_pos[i]  # 要平倉的數
                if abs(pos_tm2_pos[i]) < num[i]:
                    position[i] = 0
                if num[i] <= 0:
                    position[i] = pos_tm2_pos[i]
                else:
                    position[i] = what_position[i] * (abs(pos_tm2_pos[i]) -num[i])
            else:
                if abs(pos_tm2_pos[i]) < num_tm1_pos[i]:
                    position[i] = 0
                else:
                    position[i] = what_position[i] * (abs(pos_tm2_pos[i]) -num_tm1_pos[i])
        else:
            position[i] =a_tm1_pos[i] * num_tm1_pos[i]
    # print("pos:", position, sep="")
    return position
def pos_act(a_tm1_act, num_tm1_act, pos_tm1_act, a_tm2_act, num_tm2_act, pos_tm2_act):
    position=np.zeros((arglist.batch_size,1))
    for i in range(len(position)):
        if pos_tm2_act[i]==0:
            if a_tm1_act[i]==0:
                position[i]=0
            else:
                position[i]=a_tm1_act[i]*num_tm1_act[i]
        else:
            if a_tm1_act[i] == 0:
                position[i]=pos_tm2_act[i]
            else:
                if pos_tm2_act[i]+a_tm1_act[i]*num_tm1_act[i]>10:
                    position[i]=10
                elif pos_tm2_act[i]+a_tm1_act[i]*num_tm1_act[i]<-10:
                    position[i]=-10
                else:
                    position[i]=pos_tm2_act[i]+a_tm1_act[i]*num_tm1_act[i]
    # print("act:",position,sep="")
    return position
def profit(pos_tm1, p_t, p_tm1):
    profit = pos_tm1*(p_t-p_tm1)/p_t
    profit * 100
    return profit

if __name__ == '__main__':
    arglist = parse_args()
    arglist2 = parse_args2()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    g1 = tf.Graph()
    g2 = tf.Graph()
    sess = tf.Session(graph=g1, config=session_conf)
    sess2 = tf.Session(graph=g2, config=session_conf)
    model_act = Model_act(arglist=arglist, sess=sess)
    model_pos = Model_pos(arglist=arglist, sess2=sess2)
    sess.run(tf.global_variables())
    sess2.run(tf.global_variables())
    # save_path = model_act.saver.save(model_act.sess, "0104"+"_sess1.ckpt")
    # save_path = model_pos.saver.save(model_pos.sess2, "0104"+"_sess2.ckpt")
# pretrain by component list
    a_loss_all_eps_act, a_loss_all_eps_pos = [], []
    num_loss_all_eps_act, num_loss_all_eps_pos = [], []
    ############################################################################

    ##########################################################################
    for pre2 in range(arglist.pre2_epochs):
        component_list = ["MSFT", "AAPL", "AMZN", "FB", "JPM", "GOOG", "GOOGL", "JNJ", "V"]
        num_loss_act_each_ep, num_loss_pos_each_ep = 0, 0
        for j in range(len(component_list) - 1, -1, -1):
            # print(per_198.shape,p_t.shape)
            train_per_198, train_p_t, train_p_tm200, train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1 = download_data(
                component_list[j])
            train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1))
            for i_episode in range(arglist.n_epochs):
                for i in range(len(train_per_198)-arglist.batch_size):#+1-1
                    s, s_, b_train_p_t, b_train_p_tm200,b_train_p_tm1 = batch_train(
                        train_per_198, train_p_t, train_p_tm200,train_p_tm1, arglist=arglist, k=i)
                    b_train_p_t=b_train_p_t[:,np.newaxis]
                    b_train_p_tm1 = b_train_p_tm1[:, np.newaxis]
                    per_tm1=s[:,-1]
                    per_tm1=per_tm1[:,np.newaxis]
                    if i ==0:
                        pred_tm1_act=per_tm1.copy()
                        pred_tm1_pos =per_tm1.copy()
                    train_a_tm2_act, train_num_tm2_act, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_act, train_pos_tm2_pos = train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos
                    #act_part
                    ones = np.ones((arglist.batch_size, 1))
                    a,b=model_act.get_qacts_and_num(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    train_a_tm1_act, train_num_tm1_act = get_a_and_num(a, b)
                    train_pos_tm1_act = pos_act(train_a_tm1_act, train_num_tm1_act, train_pos_tm1_act, train_a_tm2_act, train_num_tm2_act, train_pos_tm2_act)
                    train_r_tm1_act = reward_act(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_act, train_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    q_a_act_=q_a_act_[:,np.newaxis]
                    train_num_loss_act = model_act.pretrain_2nd(s=s,s_=s_,a_tm1_act=train_a_tm1_act,ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    pred_tm1_act = model_act.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_act)

                    #pos_part
                    a2,b2=model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    train_a_tm1_pos, train_num_tm1_pos = get_a_and_num(a2,b2)

                    train_pos_tm1_pos = pos_pos(train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_pos, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_pos)
                    train_r_tm1_pos = reward_pos(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_pos, train_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    q_a_pos_=q_a_pos_[:,np.newaxis]
                    train_num_loss_pos = model_pos.pretrain_2nd(s=s,s_=s_,a_tm1_pos=train_a_tm1_pos,ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)
                    ################################
                    num_loss_act_each_ep += train_num_loss_act/arglist.batch_size
                    num_loss_pos_each_ep += train_num_loss_pos/arglist.batch_size
                    print(pre2,train_a_tm1_act[-1], train_num_tm1_act[-1],train_a_tm1_pos[-1], train_num_tm1_pos[-1],sep=",  ",end="\n")
        num_loss_all_eps_act.append(num_loss_act_each_ep)
        num_loss_all_eps_pos.append(num_loss_pos_each_ep)
    plt.figure()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    subplt(2, 1, 1, num_loss_all_eps_act, "epochs", "loss", "actmodel_pretrain_numloss")
    subplt(2, 1, 2, num_loss_all_eps_pos, "epochs", "loss", "posmodel_pretrain_numloss")
    plt.savefig("component_pretrain2_loss")
    plt.close()

    ###########################################################################
    for pre1 in range(arglist.pre1_epochs):
        component_list = ["MSFT","AAPL", "AMZN", "FB", "JPM", "GOOG", "GOOGL", "JNJ", "V"]
        a_loss_act_each_ep, a_loss_pos_each_ep = 0, 0
        for j in range(len(component_list) - 1, -1, -1):
            # print(per_198.shape,p_t.shape)
            train_per_198, train_p_t, train_p_tm200,train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1= download_data(component_list[j])
            train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros((arglist.batch_size,1)), np.zeros((arglist.batch_size,1)), np.zeros((arglist.batch_size,1)), np.zeros((arglist.batch_size,1)), np.zeros((arglist.batch_size,1)), np.zeros((arglist.batch_size,1))
            for i_episode in range(arglist.n_epochs):
                for i in range(len(train_per_198)-arglist.batch_size):#+1-1
                    s, s_, b_train_p_t, b_train_p_tm200,b_train_p_tm1 = batch_train(
                        train_per_198, train_p_t, train_p_tm200,train_p_tm1, arglist=arglist, k=i)
                    b_train_p_t=b_train_p_t[:,np.newaxis]
                    b_train_p_tm1 = b_train_p_tm1[:, np.newaxis]
                    per_tm1=s[:,-1]
                    per_tm1=per_tm1[:,np.newaxis]
                    if i ==0:
                        pred_tm1_act=per_tm1.copy()
                        pred_tm1_pos =per_tm1.copy()
                    train_a_tm2_act, train_num_tm2_act, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_act, train_pos_tm2_pos = train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos
                    #act_part
                    ones = np.ones((arglist.batch_size, 1))
                    a,b=model_act.get_qacts_and_num(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    train_a_tm1_act, train_num_tm1_act = get_a_and_num(a, b)
                    train_pos_tm1_act = pos_act(train_a_tm1_act, train_num_tm1_act, train_pos_tm1_act, train_a_tm2_act, train_num_tm2_act, train_pos_tm2_act)
                    train_r_tm1_act = reward_act(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_act, train_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    q_a_act_=q_a_act_[:,np.newaxis]
                    train_act_loss_act = model_act.pretrain_1st(r=train_r_tm1_act, s=s,s_=s_, q_=q_a_act_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    pred_tm1_act = model_act.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_act)

                    #pos_part
                    a2,b2=model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    train_a_tm1_pos, train_num_tm1_pos = get_a_and_num(a2,b2)

                    train_pos_tm1_pos = pos_pos(train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_pos, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_pos)
                    train_r_tm1_pos = reward_pos(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_pos, train_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    q_a_pos_=q_a_pos_[:,np.newaxis]
                    train_act_loss_pos = model_pos.pretrain_1st(r=train_r_tm1_pos, s=s,s_=s_, q_=q_a_pos_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)
                    print(pre1,train_a_tm1_act[-1], train_num_tm1_act[-1],train_a_tm1_pos[-1], train_num_tm1_pos[-1],sep=",  ",end="\n")
                    ##########################################################################################
                    #################
                    a_loss_act_each_ep += train_act_loss_act/arglist.batch_size
                    a_loss_pos_each_ep += train_act_loss_pos/arglist.batch_size
                    #########################################################################################
        a_loss_all_eps_act.append(a_loss_act_each_ep)
        a_loss_all_eps_pos.append(a_loss_pos_each_ep)
    plt.figure()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    subplt(2, 1, 1, a_loss_all_eps_act, "epochs", "loss", "actmodel_pretrain_actloss")
    subplt(2, 1, 2, a_loss_all_eps_pos, "epochs", "loss", "posmodel_pretrain_actloss")
    plt.savefig("component_pretrain1_loss")
    plt.close()

    plt_save(a_loss_all_eps_act,a_loss_all_eps_pos, "epochs","loss" , "pretrain act loss")
    ###########################################################################
    a_loss_all_eps_act, a_loss_all_eps_pos = [], []
    num_loss_all_eps_act, num_loss_all_eps_pos = [], []
    total_profit_all_ep_pos, total_profit_all_ep_act = [], []
    for main in range(arglist.main_epochs):
        component_list = ["MSFT", "AAPL", "AMZN", "FB", "JPM", "GOOG", "GOOGL", "JNJ", "V"]
        a_loss_act_each_ep, a_loss_pos_each_ep = 0, 0
        num_loss_act_each_ep, num_loss_pos_each_ep = 0, 0
        total_profit_each_ep_pos, total_profit_each_ep_act = 0, 0
        for j in range(len(component_list) - 1, -1, -1):
            if main == arglist.main_epochs-1:
                a_act, num_act, a_pos, num_pos = [], [], [], []
                accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list = [],[],[]
                accumulated_profit, accumulated_profit_act, accumulated_profit_pos = 0,0,0
            # print(per_198.shape,p_t.shape)
            train_per_198, train_p_t, train_p_tm200, train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1 = download_data(
                component_list[j])
            train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1))
            test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos = np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
                (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1))
            for i_episode in range(arglist.n_epochs):
                for i in range(len(train_per_198)-arglist.batch_size):#+1-1
                    s, s_, b_train_p_t, b_train_p_tm200,b_train_p_tm1 = batch_train(
                        train_per_198, train_p_t, train_p_tm200,train_p_tm1, arglist=arglist, k=i)
                    b_train_p_t=b_train_p_t[:,np.newaxis]
                    b_train_p_tm1 = b_train_p_tm1[:, np.newaxis]
                    per_tm1=s[:,-1]
                    per_tm1=per_tm1[:,np.newaxis]
                    if i ==0:
                        pred_tm1_act=per_tm1.copy()
                        pred_tm1_pos =per_tm1.copy()
                    train_a_tm2_act, train_num_tm2_act, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_act, train_pos_tm2_pos = train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos
                    #act_part
                    ones = np.ones((arglist.batch_size, 1))
                    a,b=model_act.get_qacts_and_num(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    train_a_tm1_act, train_num_tm1_act = get_a_and_num(a, b)
                    train_pos_tm1_act = pos_act(train_a_tm1_act, train_num_tm1_act, train_pos_tm1_act, train_a_tm2_act, train_num_tm2_act, train_pos_tm2_act)
                    train_r_tm1_act = reward_act(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_act, train_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    q_a_act_=q_a_act_[:,np.newaxis]
                    train_act_loss_act, train_num_loss_act = model_act.main_train(r=train_r_tm1_act, s=s,s_=s_, act_q_=q_a_act_,a_tm1_act=train_a_tm1_act,ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    # print(a_tm1_act.shape,num_tm1_act.shape)
                    pred_tm1_act = model_act.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_act)

                    #pos_part
                    a2,b2=model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    train_a_tm1_pos, train_num_tm1_pos = get_a_and_num(a2,b2)

                    train_pos_tm1_pos = pos_pos(train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_pos, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_pos)
                    train_r_tm1_pos = reward_pos(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_pos, train_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    q_a_pos_=q_a_pos_[:,np.newaxis]
                    train_act_loss_pos, train_num_loss_pos = model_pos.main_train(r=train_r_tm1_pos, s=s,s_=s_, act_q_=q_a_pos_, a_tm1_pos=train_a_tm1_pos, ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)

                    pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)

                    print(main, train_a_tm1_act[-1], train_num_tm1_act[-1], train_a_tm1_pos[-1], train_num_tm1_pos[-1],
                          sep=",  ", end="\n")

                    profit_i_act = profit(train_pos_tm1_act[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
                    profit_i_pos = profit(train_pos_tm1_pos[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
                    p_tp1_i = profit(10, b_train_p_t[-1], b_train_p_tm1[-1])


                    ####################################################
                    a_loss_act_each_ep+=train_act_loss_act/arglist.batch_size
                    a_loss_pos_each_ep+=train_act_loss_pos/arglist.batch_size
                    num_loss_act_each_ep += train_num_loss_act/arglist.batch_size
                    num_loss_pos_each_ep += train_num_loss_pos/arglist.batch_size
                    ####################################################
                    total_profit_each_ep_act+=profit_i_act.copy()
                    total_profit_each_ep_pos+=profit_i_pos.copy()
                    ##########################################################################################
                    if main == arglist.main_epochs-1:
                        a_act.append(train_a_tm1_act[-1])
                        num_act.append(train_num_tm1_act[-1])
                        a_pos.append(train_a_tm1_pos[-1])
                        num_pos.append(train_num_tm1_pos[-1])
                        accumulated_profit_pos += profit_i_pos.copy()
                        accumulated_profit_act += profit_i_act.copy()
                        accumulated_profit += p_tp1_i.copy()
                        accumulated_profit_list.append(accumulated_profit.copy())
                        accumulated_profit_act_list.append(accumulated_profit_act.copy())
                        accumulated_profit_pos_list.append(accumulated_profit_pos.copy())



                for i in range(len(test_per_198)-arglist.batch_size):
                    ones = np.ones((arglist.batch_size, 1))
                    s, s_, b_test_p_t, b_test_p_tm200, b_test_p_tm1, = batch_test(
                        test_per_198, test_p_t, test_p_tm200, test_p_tm1, arglist=arglist, k=i)
                    # print(s.shape)
                    per_tm1 = s[:, -1]
                    per_tm1 = per_tm1[:, np.newaxis]
                    if i == 0:
                        pred_tm1_act = per_tm1.copy()
                        pred_tm1_pos = per_tm1.copy()
                    b_test_p_t = b_test_p_t[:, np.newaxis]
                    b_test_p_tm1 = b_test_p_tm1[:, np.newaxis]
                    test_a_tm2_act, test_num_tm2_act, test_a_tm2_pos, test_num_tm2_pos, test_pos_tm2_act, test_pos_tm2_pos = test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos
                    # act_part
                    a, b = model_act.get_qacts_and_num(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    test_a_tm1_act, test_num_tm1_act = get_a_and_num(a, b)
                    test_pos_tm1_act = pos_act(test_a_tm1_act, test_num_tm1_act, test_pos_tm1_act, test_a_tm2_act,
                                               test_num_tm2_act, test_pos_tm2_act)
                    test_r_tm1_act = reward_act(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_act, test_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    q_a_act_ = q_a_act_[:, np.newaxis]
                    # print(test_r_tm1_act.shape, s_.shape, q_a_act_.shape, q_num_act_.shape)
                    test_act_loss_act = model_act.pretrain_1st(r=test_r_tm1_act, s=s,s_=s_, q_=q_a_act_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    test_num_loss_act = model_act.pretrain_2nd(s=s,s_=s_,a_tm1_act=test_a_tm1_act,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    test_act_loss_act, test_num_loss_act = model_act.main_train(r=test_r_tm1_act, s=s,s_=s_, act_q_=q_a_act_,a_tm1_act=test_a_tm1_act,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    # print(a_tm1_act.shape,num_tm1_act.shape)
                    pred_tm1_act=model_act.get_per(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
                    # pos_part
                    a2, b2 = model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    test_a_tm1_pos, test_num_tm1_pos = get_a_and_num(a2, b2)
                    test_pos_tm1_pos = pos_pos(test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_pos, test_a_tm2_pos,
                                               test_num_tm2_pos, test_pos_tm2_pos)
                    test_r_tm1_pos = reward_pos(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_pos, test_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    q_a_pos_ = q_a_pos_[:, np.newaxis]
                    test_act_loss_pos = model_pos.pretrain_1st(r=test_r_tm1_pos, s=s,s_=s_, q_=q_a_pos_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    test_num_loss_pos = model_pos.pretrain_2nd(s=s,s_=s_,a_tm1_pos=test_a_tm1_pos, ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    test_act_loss_pos, test_num_loss_pos = model_pos.main_train(r=test_r_tm1_pos, s=s,s_=s_, act_q_=q_a_pos_, a_tm1_pos=test_a_tm1_pos,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
                    pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)

                    profit_i_act=profit(test_pos_tm1_act[-1][0],b_test_p_t[-1],b_test_p_tm1[-1])
                    profit_i_pos= profit(test_pos_tm1_pos[-1][0],b_test_p_t[-1],b_test_p_tm1[-1])
                    p_tp1_i = profit(10,b_test_p_t[-1],b_test_p_tm1[-1])
                    print(main, test_a_tm1_act[-1], test_num_tm1_act[-1], test_a_tm1_pos[-1], test_num_tm1_pos[-1],
                          sep=",  ", end="\n")
                    ####################################################
                    ####################################################
                    a_loss_act_each_ep += test_act_loss_act/arglist.batch_size
                    a_loss_pos_each_ep += test_act_loss_pos/arglist.batch_size
                    num_loss_act_each_ep += test_num_loss_act/arglist.batch_size
                    num_loss_pos_each_ep += test_num_loss_pos/arglist.batch_size
                    ####################################################
                    total_profit_each_ep_act += profit_i_act.copy()
                    total_profit_each_ep_pos += profit_i_pos.copy()
                    ##########################################################################################
                    if main == arglist.main_epochs - 1:
                        a_act.append(test_a_tm1_act[-1])
                        num_act.append(test_num_tm1_act[-1])
                        a_pos.append(test_a_tm1_pos[-1])
                        num_pos.append(test_num_tm1_pos[-1])
                        accumulated_profit_pos += profit_i_pos.copy()
                        accumulated_profit_act += profit_i_act.copy()
                        accumulated_profit += p_tp1_i.copy()
                        accumulated_profit_list.append(accumulated_profit.copy())
                        accumulated_profit_act_list.append(accumulated_profit_act.copy())
                        accumulated_profit_pos_list.append(accumulated_profit_pos.copy())
                    ##########################################################################################

            if main == arglist.main_epochs-1:
                plt_total_profit(accumulated_profit_list,accumulated_profit_act_list,accumulated_profit_pos_list, str(9-j)+"_"+component_list[j]+"_accumulated profit(%)")
                plt_num_act(a_act, num_act, "time", "num", str(9-j)+"_"+component_list[j]+"_act_policy")
                plt_num_act(a_pos, num_pos, "time", "num", str(9 - j) + "_" + component_list[j] + "_pos_policy")
        num_loss_all_eps_act.append(num_loss_act_each_ep)
        num_loss_all_eps_pos.append(num_loss_pos_each_ep)
        a_loss_all_eps_act.append(a_loss_act_each_ep)
        a_loss_all_eps_pos.append(a_loss_pos_each_ep)
        total_profit_all_ep_pos.append(total_profit_each_ep_pos)
        total_profit_all_ep_act.append(total_profit_each_ep_act)
    plt.figure()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    subplt(2, 2, 1, a_loss_all_eps_act, "epochs", "loss", "actmodel_maintrain_actloss")
    subplt(2, 2, 2, a_loss_all_eps_pos, "epochs", "loss", "posmodel_maintrain_actloss")
    subplt(2, 2, 3, num_loss_all_eps_act, "epochs", "loss", "actmodel_maintrain_numloss")
    subplt(2, 2, 4, num_loss_all_eps_pos, "epochs", "loss", "posmodel_maintrain_numloss")
    plt.savefig("component_maintrain_loss")
    plt.close()
    plt_save(total_profit_all_ep_act, total_profit_all_ep_pos, "time", "total profit(%)", "component_total_profit")
    # for test in range(arglist.test_epochs):
    #     component_list = ["MSFT", "AAPL", "AMZN", "FB", "JPM", "GOOG", "GOOGL", "JNJ", "V"]
    #     a_loss_act_each_ep, a_loss_pos_each_ep = 0, 0
    #     num_loss_act_each_ep, num_loss_pos_each_ep = 0, 0
    #     total_profit_each_ep_pos, total_profit_each_ep_act = 0, 0
    #     for j in range(len(component_list) - 1, -1, -1):
    #         if test == arglist.test_epochs-1:
    #             a_act, num_act, a_pos, num_pos = [], [], [], []
    #             accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list = [],[],[]
    #             accumulated_profit, accumulated_profit_act, accumulated_profit_pos = 0,0,0
    #         # print(per_198.shape,p_t.shape)
    #         train_per_198, train_p_t, train_p_tm200, train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1 = download_data(
    #             component_list[j])
    #         train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1))
    #         test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos = np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1)), np.zeros(
    #             (arglist.batch_size, 1)), np.zeros((arglist.batch_size, 1))
    #         for i_episode in range(arglist.n_epochs):
    #             for i in range(len(train_per_198)-arglist.batch_size):#+1-1
    #                 s, s_, b_train_p_t, b_train_p_tm200,b_train_p_tm1 = batch_train(
    #                     train_per_198, train_p_t, train_p_tm200,train_p_tm1, arglist=arglist, k=i)
    #                 b_train_p_t=b_train_p_t[:,np.newaxis]
    #                 b_train_p_tm1 = b_train_p_tm1[:, np.newaxis]
    #                 per_tm1=s[:,-1]
    #                 per_tm1=per_tm1[:,np.newaxis]
    #                 if i ==0:
    #                     pred_tm1_act=per_tm1.copy()
    #                     pred_tm1_pos =per_tm1.copy()
    #                 train_a_tm2_act, train_num_tm2_act, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_act, train_pos_tm2_pos = train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos
    #                 #act_part
    #                 ones = np.ones((arglist.batch_size, 1))
    #                 a,b=model_act.get_qacts_and_num(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 train_a_tm1_act, train_num_tm1_act = get_a_and_num(a, b)
    #                 train_pos_tm1_act = pos_act(train_a_tm1_act, train_num_tm1_act, train_pos_tm1_act, train_a_tm2_act, train_num_tm2_act, train_pos_tm2_act)
    #                 train_r_tm1_act = reward_act(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_act, train_a_tm1_act)
    #                 q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_, p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 q_a_act_=q_a_act_[:,np.newaxis]
    #                 train_act_loss_act, train_num_loss_act = model_act.main_train(r=train_r_tm1_act, s=s,s_=s_, act_q_=q_a_act_,a_tm1_act=train_a_tm1_act,ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 # print(a_tm1_act.shape,num_tm1_act.shape)
    #                 pred_tm1_act = model_act.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
    #                                                  pred_tm1=pred_tm1_act)
    #
    #                 #pos_part
    #                 a2,b2=model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 train_a_tm1_pos, train_num_tm1_pos = get_a_and_num(a2,b2)
    #
    #                 train_pos_tm1_pos = pos_pos(train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_pos, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_pos)
    #                 train_r_tm1_pos = reward_pos(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_pos, train_pos_tm1_pos)
    #                 q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_train_p_t,p_tm1=b_train_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 q_a_pos_=q_a_pos_[:,np.newaxis]
    #                 train_act_loss_pos, train_num_loss_pos = model_pos.main_train(r=train_r_tm1_pos, s=s,s_=s_, act_q_=q_a_pos_, a_tm1_pos=train_a_tm1_pos, ones=ones,p_t=b_train_p_t, p_tm1=b_train_p_tm1, p_tm200=b_train_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #
    #                 pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
    #                                                  pred_tm1=pred_tm1_pos)
    #
    #                 print(test, train_a_tm1_act[-1], train_num_tm1_act[-1], train_a_tm1_pos[-1], train_num_tm1_pos[-1],
    #                       sep=",  ", end="\n")
    #
    #                 profit_i_act = profit(train_pos_tm1_act[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
    #                 profit_i_pos = profit(train_pos_tm1_pos[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
    #                 p_tp1_i = profit(10, b_train_p_t[-1], b_train_p_tm1[-1])
    #
    #
    #                 ####################################################
    #                 a_loss_act_each_ep+=train_act_loss_act/arglist.batch_size
    #                 a_loss_pos_each_ep+=train_act_loss_pos/arglist.batch_size
    #                 num_loss_act_each_ep += train_num_loss_act/arglist.batch_size
    #                 num_loss_pos_each_ep += train_num_loss_pos/arglist.batch_size
    #                 ####################################################
    #                 total_profit_each_ep_act+=profit_i_act.copy()
    #                 total_profit_each_ep_pos+=profit_i_pos.copy()
    #                 ##########################################################################################
    #                 if test == arglist.test_epochs-1:
    #                     a_act.append(train_a_tm1_act[-1])
    #                     num_act.append(train_num_tm1_act[-1])
    #                     a_pos.append(train_a_tm1_pos[-1])
    #                     num_pos.append(train_num_tm1_pos[-1])
    #                     accumulated_profit_pos += profit_i_pos.copy()
    #                     accumulated_profit_act += profit_i_act.copy()
    #                     accumulated_profit += p_tp1_i.copy()
    #                     accumulated_profit_list.append(accumulated_profit.copy())
    #                     accumulated_profit_act_list.append(accumulated_profit_act.copy())
    #                     accumulated_profit_pos_list.append(accumulated_profit_pos.copy())
    #
    #
    #
    #             for i in range(len(test_per_198)-arglist.batch_size):
    #                 ones = np.ones((arglist.batch_size, 1))
    #                 s, s_, b_test_p_t, b_test_p_tm200, b_test_p_tm1, = batch_test(
    #                     test_per_198, test_p_t, test_p_tm200, test_p_tm1, arglist=arglist, k=i)
    #                 # print(s.shape)
    #                 per_tm1 = s[:, -1]
    #                 per_tm1 = per_tm1[:, np.newaxis]
    #                 if i == 0:
    #                     pred_tm1_act = per_tm1.copy()
    #                     pred_tm1_pos = per_tm1.copy()
    #                 b_test_p_t = b_test_p_t[:, np.newaxis]
    #                 b_test_p_tm1 = b_test_p_tm1[:, np.newaxis]
    #                 test_a_tm2_act, test_num_tm2_act, test_a_tm2_pos, test_num_tm2_pos, test_pos_tm2_act, test_pos_tm2_pos = test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos
    #                 # act_part
    #                 a, b = model_act.get_qacts_and_num(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 test_a_tm1_act, test_num_tm1_act = get_a_and_num(a, b)
    #                 test_pos_tm1_act = pos_act(test_a_tm1_act, test_num_tm1_act, test_pos_tm1_act, test_a_tm2_act,
    #                                            test_num_tm2_act, test_pos_tm2_act)
    #                 test_r_tm1_act = reward_act(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_act, test_a_tm1_act)
    #                 q_a_act_, q_num_act_ = model_act.get_q_score(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 q_a_act_ = q_a_act_[:, np.newaxis]
    #                 # print(test_r_tm1_act.shape, s_.shape, q_a_act_.shape, q_num_act_.shape)
    #                 test_act_loss_act = model_act.pretrain_1st(r=test_r_tm1_act, s=s,s_=s_, q_=q_a_act_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 test_num_loss_act = model_act.pretrain_2nd(s=s,s_=s_,a_tm1_act=test_a_tm1_act,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 test_act_loss_act, test_num_loss_act = model_act.main_train(r=test_r_tm1_act, s=s,s_=s_, act_q_=q_a_act_,a_tm1_act=test_a_tm1_act,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 # print(a_tm1_act.shape,num_tm1_act.shape)
    #                 pred_tm1_act=model_act.get_per(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_act)
    #                 # pos_part
    #                 a2, b2 = model_pos.get_qacts_and_num(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 test_a_tm1_pos, test_num_tm1_pos = get_a_and_num(a2, b2)
    #                 test_pos_tm1_pos = pos_pos(test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_pos, test_a_tm2_pos,
    #                                            test_num_tm2_pos, test_pos_tm2_pos)
    #                 test_r_tm1_pos = reward_pos(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_pos, test_pos_tm1_pos)
    #                 q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s,s_=s_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 q_a_pos_ = q_a_pos_[:, np.newaxis]
    #                 test_act_loss_pos = model_pos.pretrain_1st(r=test_r_tm1_pos, s=s,s_=s_, q_=q_a_pos_,p_t=b_test_p_t,p_tm1=b_test_p_tm1,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 test_num_loss_pos = model_pos.pretrain_2nd(s=s,s_=s_,a_tm1_pos=test_a_tm1_pos, ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 test_act_loss_pos, test_num_loss_pos = model_pos.main_train(r=test_r_tm1_pos, s=s,s_=s_, act_q_=q_a_pos_, a_tm1_pos=test_a_tm1_pos,ones=ones,p_t=b_test_p_t, p_tm1=b_test_p_tm1, p_tm200=b_test_p_tm200,per_tm1=per_tm1,pred_tm1=pred_tm1_pos)
    #                 pred_tm1_pos = model_pos.get_per(s=s,s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1, per_tm1=per_tm1,
    #                                                  pred_tm1=pred_tm1_pos)
    #
    #                 profit_i_act=profit(test_pos_tm1_act[-1][0],b_test_p_t[-1],b_test_p_tm1[-1])
    #                 profit_i_pos= profit(test_pos_tm1_pos[-1][0],b_test_p_t[-1],b_test_p_tm1[-1])
    #                 p_tp1_i = profit(10,b_test_p_t[-1],b_test_p_tm1[-1])
    #                 print(test, test_a_tm1_act[-1], test_num_tm1_act[-1], test_a_tm1_pos[-1], test_num_tm1_pos[-1],
    #                       sep=",  ", end="\n")
    #                 ####################################################
    #                 ####################################################
    #                 a_loss_act_each_ep += test_act_loss_act/arglist.batch_size
    #                 a_loss_pos_each_ep += test_act_loss_pos/arglist.batch_size
    #                 num_loss_act_each_ep += test_num_loss_act/arglist.batch_size
    #                 num_loss_pos_each_ep += test_num_loss_pos/arglist.batch_size
    #                 ####################################################
    #                 total_profit_each_ep_act += profit_i_act.copy()
    #                 total_profit_each_ep_pos += profit_i_pos.copy()
    #                 ##########################################################################################
    #                 if test == arglist.test_epochs - 1:
    #                     a_act.append(test_a_tm1_act[-1])
    #                     num_act.append(test_num_tm1_act[-1])
    #                     a_pos.append(test_a_tm1_pos[-1])
    #                     num_pos.append(test_num_tm1_pos[-1])
    #                     accumulated_profit_pos += profit_i_pos.copy()
    #                     accumulated_profit_act += profit_i_act.copy()
    #                     accumulated_profit += p_tp1_i.copy()
    #                     accumulated_profit_list.append(accumulated_profit.copy())
    #                     accumulated_profit_act_list.append(accumulated_profit_act.copy())
    #                     accumulated_profit_pos_list.append(accumulated_profit_pos.copy())
    #                 ##########################################################################################
    #
    #         if test == arglist.test_epochs-1:
    #             plt_total_profit(accumulated_profit_list,accumulated_profit_act_list,accumulated_profit_pos_list, str(9-j)+"_"+component_list[j]+"_accumulated profit(%)")
    #             plt_num_act(a_act, num_act, "time", "num", str(9-j)+"_"+component_list[j]+"_act_policy")
    #             plt_num_act(a_pos, num_pos, "time", "num", str(9 - j) + "_" + component_list[j] + "_pos_policy")
    save_path = model_act.saver.save(model_act.sess, "_pretrain_sess.ckpt")
    save_path = model_pos.saver.save(model_pos.sess2, "_pretrIn_sess2.ckpt")



# main train indexes
    a_loss_all_eps_act, a_loss_all_eps_pos = [], []
    num_loss_all_eps_act, num_loss_all_eps_pos = [], []
    ##############################################################################

##############################################################################################

    #######################################################################

    ############################################################################
    a_loss_all_eps_act, a_loss_all_eps_pos = [], []
    num_loss_all_eps_act, num_loss_all_eps_pos = [], []
    total_profit_all_ep_pos, total_profit_all_ep_act = [], []
    for main in range(arglist2.main_epochs):
        index_list = ["^GSPC"]
        a_loss_act_each_ep, a_loss_pos_each_ep = 0, 0
        num_loss_act_each_ep, num_loss_pos_each_ep = 0, 0
        total_profit_each_ep_pos, total_profit_each_ep_act = 0, 0
        for j in range(len(index_list) - 1, -1, -1):
            if main == arglist2.main_epochs - 1:
                a_act, num_act, a_pos, num_pos = [], [], [], []
                accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list = [], [], []
                accumulated_profit, accumulated_profit_act, accumulated_profit_pos = 0, 0, 0
            # print(per_198.shape,p_t.shape)
            train_per_198, train_p_t, train_p_tm200, train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1 = download_data(
                index_list[j])
            train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1))
            test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos = np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1))
            for i_episode in range(arglist2.n_epochs):
                for i in range(len(train_per_198) - arglist2.batch_size):  # +1-1
                    s, s_, b_train_p_t, b_train_p_tm200, b_train_p_tm1 = batch_train(
                        train_per_198, train_p_t, train_p_tm200, train_p_tm1, arglist=arglist2, k=i)
                    b_train_p_t = b_train_p_t[:, np.newaxis]
                    b_train_p_tm1 = b_train_p_tm1[:, np.newaxis]
                    per_tm1 = s[:, -1]
                    per_tm1 = per_tm1[:, np.newaxis]
                    if i == 0:
                        pred_tm1_act = per_tm1.copy()
                        pred_tm1_pos = per_tm1.copy()
                    train_a_tm2_act, train_num_tm2_act, train_a_tm2_pos, train_num_tm2_pos, train_pos_tm2_act, train_pos_tm2_pos = train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos
                    # act_part
                    ones = np.ones((arglist2.batch_size, 1))
                    a, b = model_act.get_qacts_and_num(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                       per_tm1=per_tm1,
                                                       pred_tm1=pred_tm1_act)
                    train_a_tm1_act, train_num_tm1_act = get_a_and_num(a, b)
                    train_pos_tm1_act = pos_act(train_a_tm1_act, train_num_tm1_act, train_pos_tm1_act, train_a_tm2_act,
                                                train_num_tm2_act, train_pos_tm2_act)
                    train_r_tm1_act = reward_act(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_act,
                                                 train_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                                 per_tm1=per_tm1, pred_tm1=pred_tm1_act)
                    q_a_act_ = q_a_act_[:, np.newaxis]
                    train_act_loss_act, train_num_loss_act = model_act.main_train(r=train_r_tm1_act, s=s, s_=s_,
                                                                                  act_q_=q_a_act_,
                                                                                  a_tm1_act=train_a_tm1_act, ones=ones,
                                                                                  p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                                                  p_tm200=b_train_p_tm200,
                                                                                  per_tm1=per_tm1,
                                                                                  pred_tm1=pred_tm1_act)
                    # print(a_tm1_act.shape,num_tm1_act.shape)
                    pred_tm1_act = model_act.get_per(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_act)

                    # pos_part
                    a2, b2 = model_pos.get_qacts_and_num(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                         per_tm1=per_tm1,
                                                         pred_tm1=pred_tm1_pos)
                    train_a_tm1_pos, train_num_tm1_pos = get_a_and_num(a2, b2)

                    train_pos_tm1_pos = pos_pos(train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_pos, train_a_tm2_pos,
                                                train_num_tm2_pos, train_pos_tm2_pos)
                    train_r_tm1_pos = reward_pos(b_train_p_t, b_train_p_tm1, b_train_p_tm200, train_num_tm1_pos,
                                                 train_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                                 per_tm1=per_tm1, pred_tm1=pred_tm1_pos)
                    q_a_pos_ = q_a_pos_[:, np.newaxis]
                    train_act_loss_pos, train_num_loss_pos = model_pos.main_train(r=train_r_tm1_pos, s=s, s_=s_,
                                                                                  act_q_=q_a_pos_,
                                                                                  a_tm1_pos=train_a_tm1_pos, ones=ones,
                                                                                  p_t=b_train_p_t, p_tm1=b_train_p_tm1,
                                                                                  p_tm200=b_train_p_tm200,
                                                                                  per_tm1=per_tm1,
                                                                                  pred_tm1=pred_tm1_pos)

                    pred_tm1_pos = model_pos.get_per(s=s, s_=s_, p_t=b_train_p_t, p_tm1=b_train_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)

                    profit_i_act = profit(train_pos_tm1_act[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
                    profit_i_pos = profit(train_pos_tm1_pos[-1][0], b_train_p_t[-1], b_train_p_tm1[-1])
                    p_tp1_i = profit(10, b_train_p_t[-1], b_train_p_tm1[-1])

                    ####################################################
                    a_loss_act_each_ep += train_act_loss_act / arglist2.batch_size
                    a_loss_pos_each_ep += train_act_loss_pos / arglist2.batch_size
                    num_loss_act_each_ep += train_num_loss_act / arglist2.batch_size
                    num_loss_pos_each_ep += train_num_loss_pos / arglist2.batch_size
                    ####################################################
                    total_profit_each_ep_act += profit_i_act.copy()
                    total_profit_each_ep_pos += profit_i_pos.copy()
                    ##########################################################################################
                    if main == arglist2.main_epochs - 1:
                        a_act.append(train_a_tm1_act[-1])
                        num_act.append(train_num_tm1_act[-1])
                        a_pos.append(train_a_tm1_pos[-1])
                        num_pos.append(train_num_tm1_pos[-1])
                        accumulated_profit_pos += profit_i_pos.copy()
                        accumulated_profit_act += profit_i_act.copy()
                        accumulated_profit += p_tp1_i.copy()
                        accumulated_profit_list.append(accumulated_profit.copy())
                        accumulated_profit_act_list.append(accumulated_profit_act.copy())
                        accumulated_profit_pos_list.append(accumulated_profit_pos.copy())
                num_loss_all_eps_act.append(num_loss_act_each_ep)
                num_loss_all_eps_pos.append(num_loss_pos_each_ep)
                a_loss_all_eps_act.append(a_loss_act_each_ep)
                a_loss_all_eps_pos.append(a_loss_pos_each_ep)
                total_profit_all_ep_pos.append(total_profit_each_ep_pos)
                total_profit_all_ep_act.append(total_profit_each_ep_act)
            if main == arglist2.main_epochs - 1:
                plt_total_profit(accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list,
                                 str(9 - j) + "_" + index_list[j] + "_accumulated profit(%)")
                plt_num_act(a_act, num_act, "time", "num", str(9 - j) + "_" + index_list[j] + "_act_policy")
                plt_num_act(a_pos, num_pos, "time", "num", str(9 - j) + "_" + index_list[j] + "_pos_policy")
    plt.figure()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    subplt(2, 2, 1, a_loss_all_eps_act, "epochs", "loss", "actmodel_maintrain_actloss")
    subplt(2, 2, 2, a_loss_all_eps_pos, "epochs", "loss", "posmodel_maintrain_actloss")
    subplt(2, 2, 3, num_loss_all_eps_act, "epochs", "loss", "actmodel_maintrain_numloss")
    subplt(2, 2, 4, num_loss_all_eps_pos, "epochs", "loss", "posmodel_maintrain_numloss")
    plt.savefig("index_maintrain_loss")
    plt.close()
    plt_save(total_profit_all_ep_act, total_profit_all_ep_pos, "time", "total profit(%)", "index_total_profit")
    for test in range(arglist2.test_epochs):
        index_list = ["^GSPC"]
        for j in range(len(index_list)):
            if test == arglist2.test_epochs - 1:
                a_act, num_act, a_pos, num_pos = [], [], [], []
                accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list = [], [], []
                accumulated_profit, accumulated_profit_act, accumulated_profit_pos = 0, 0, 0
            # print(per_198.shape,p_t.shape)
            train_per_198, train_p_t, train_p_tm200, train_p_tm1, test_per_198, test_p_t, test_p_tm200, test_p_tm1 = download_data(
                index_list[j])
            train_a_tm1_act, train_num_tm1_act, train_a_tm1_pos, train_num_tm1_pos, train_pos_tm1_act, train_pos_tm1_pos = np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1))
            test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos = np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1)), np.zeros(
                (arglist2.batch_size, 1)), np.zeros((arglist2.batch_size, 1))
            for i_episode in range(arglist2.n_epochs):

                for i in range(len(test_per_198) - arglist2.batch_size):
                    ones = np.ones((arglist2.batch_size, 1))
                    s, s_, b_test_p_t, b_test_p_tm200, b_test_p_tm1, = batch_test(
                        test_per_198, test_p_t, test_p_tm200, test_p_tm1, arglist=arglist2, k=i)
                    # print(s.shape)
                    per_tm1 = s[:, -1]
                    per_tm1 = per_tm1[:, np.newaxis]
                    if i == 0:
                        pred_tm1_act = per_tm1.copy()
                        pred_tm1_pos = per_tm1.copy()
                    b_test_p_t = b_test_p_t[:, np.newaxis]
                    b_test_p_tm1 = b_test_p_tm1[:, np.newaxis]
                    test_a_tm2_act, test_num_tm2_act, test_a_tm2_pos, test_num_tm2_pos, test_pos_tm2_act, test_pos_tm2_pos = test_a_tm1_act, test_num_tm1_act, test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_act, test_pos_tm1_pos
                    # act_part
                    a, b = model_act.get_qacts_and_num(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                       pred_tm1=pred_tm1_act)
                    test_a_tm1_act, test_num_tm1_act = get_a_and_num(a, b)
                    test_pos_tm1_act = pos_act(test_a_tm1_act, test_num_tm1_act, test_pos_tm1_act, test_a_tm2_act,
                                               test_num_tm2_act, test_pos_tm2_act)
                    test_r_tm1_act = reward_act(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_act,
                                                test_a_tm1_act)
                    q_a_act_, q_num_act_ = model_act.get_q_score(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1,
                                                                 per_tm1=per_tm1, pred_tm1=pred_tm1_act)
                    q_a_act_ = q_a_act_[:, np.newaxis]
                    # print(test_r_tm1_act.shape, s_.shape, q_a_act_.shape, q_num_act_.shape)
                    test_act_loss_act = model_act.pretrain_1st(r=test_r_tm1_act, s=s, s_=s_, q_=q_a_act_,
                                                               p_t=b_test_p_t,
                                                               p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                               pred_tm1=pred_tm1_act)
                    test_num_loss_act = model_act.pretrain_2nd(s=s, s_=s_, a_tm1_act=test_a_tm1_act, ones=ones,
                                                               p_t=b_test_p_t, p_tm1=b_test_p_tm1,
                                                               p_tm200=b_test_p_tm200,
                                                               per_tm1=per_tm1, pred_tm1=pred_tm1_act)
                    test_act_loss_act, test_num_loss_act = model_act.main_train(r=test_r_tm1_act, s=s, s_=s_,
                                                                                act_q_=q_a_act_,
                                                                                a_tm1_act=test_a_tm1_act,
                                                                                ones=ones, p_t=b_test_p_t,
                                                                                p_tm1=b_test_p_tm1,
                                                                                p_tm200=b_test_p_tm200,
                                                                                per_tm1=per_tm1, pred_tm1=pred_tm1_act)
                    # print(a_tm1_act.shape,num_tm1_act.shape)
                    pred_tm1_act = model_act.get_per(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_act)
                    # pos_part
                    a2, b2 = model_pos.get_qacts_and_num(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1,
                                                         per_tm1=per_tm1,
                                                         pred_tm1=pred_tm1_pos)
                    test_a_tm1_pos, test_num_tm1_pos = get_a_and_num(a2, b2)
                    test_pos_tm1_pos = pos_pos(test_a_tm1_pos, test_num_tm1_pos, test_pos_tm1_pos, test_a_tm2_pos,
                                               test_num_tm2_pos, test_pos_tm2_pos)
                    test_r_tm1_pos = reward_pos(b_test_p_t, b_test_p_tm1, b_test_p_tm200, test_num_tm1_pos,
                                                test_pos_tm1_pos)
                    q_a_pos_, q_num_pos_ = model_pos.get_q_score(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1,
                                                                 per_tm1=per_tm1, pred_tm1=pred_tm1_pos)
                    q_a_pos_ = q_a_pos_[:, np.newaxis]
                    test_act_loss_pos = model_pos.pretrain_1st(r=test_r_tm1_pos, s=s, s_=s_, q_=q_a_pos_,
                                                               p_t=b_test_p_t,
                                                               p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                               pred_tm1=pred_tm1_pos)
                    test_num_loss_pos = model_pos.pretrain_2nd(s=s, s_=s_, a_tm1_pos=test_a_tm1_pos, ones=ones,
                                                               p_t=b_test_p_t, p_tm1=b_test_p_tm1,
                                                               p_tm200=b_test_p_tm200,
                                                               per_tm1=per_tm1, pred_tm1=pred_tm1_pos)
                    test_act_loss_pos, test_num_loss_pos = model_pos.main_train(r=test_r_tm1_pos, s=s, s_=s_,
                                                                                act_q_=q_a_pos_,
                                                                                a_tm1_pos=test_a_tm1_pos,
                                                                                ones=ones, p_t=b_test_p_t,
                                                                                p_tm1=b_test_p_tm1,
                                                                                p_tm200=b_test_p_tm200,
                                                                                per_tm1=per_tm1, pred_tm1=pred_tm1_pos)
                    pred_tm1_pos = model_pos.get_per(s=s, s_=s_, p_t=b_test_p_t, p_tm1=b_test_p_tm1, per_tm1=per_tm1,
                                                     pred_tm1=pred_tm1_pos)

                    profit_i_act = profit(test_pos_tm1_act[-1][0], b_test_p_t[-1], b_test_p_tm1[-1])
                    profit_i_pos = profit(test_pos_tm1_pos[-1][0], b_test_p_t[-1], b_test_p_tm1[-1])
                    p_tp1_i = profit(10, b_test_p_t[-1], b_test_p_tm1[-1])
                    ####################################################
                    ####################################################
                    ####################################################
                    ##########################################################################################
                    if test == arglist2.test_epochs - 1:
                        a_act.append(test_a_tm1_act[-1])
                        num_act.append(test_num_tm1_act[-1])
                        a_pos.append(test_a_tm1_pos[-1])
                        num_pos.append(test_num_tm1_pos[-1])
                        accumulated_profit_pos += profit_i_pos.copy()
                        accumulated_profit_act += profit_i_act.copy()
                        accumulated_profit += p_tp1_i.copy()
                        accumulated_profit_list.append(accumulated_profit.copy())
                        accumulated_profit_act_list.append(accumulated_profit_act.copy())
                        accumulated_profit_pos_list.append(accumulated_profit_pos.copy())
                    ##########################################################################################
            if test == arglist2.test_epochs - 1:
                plt_total_profit(accumulated_profit_list, accumulated_profit_act_list, accumulated_profit_pos_list,
                                  index_list[j] + "_accumulated profit(%)")
                plt_num_act(a_act, num_act, "time", "num",  index_list[j] + "_act_policy")
                plt_num_act(a_pos, num_pos, "time", "num",  index_list[j] + "_pos_policy")

    save_path = model_act.saver.save(model_act.sess, "_index_sess.ckpt")
    save_path = model_pos.saver.save(model_pos.sess2, "_index_sess2.ckpt")