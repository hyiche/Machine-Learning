import yfinance as yf
from pandas_datareader import data as pdr
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import datetime
import time
import os
import cv2
def plt_num_act(act, xl, yl, name):
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
    plt.legend(recs,label)
    plt.savefig(name)
    plt.close()
    # plt.show()
def plt_per(per,name,xl,yl):
    plt.figure()
    plt.title(name,y=1.05)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.plot(per, color='black',linewidth=0.5)
    plt.savefig(name)
    plt.close()
    # plt.show()
def parse_args():
    parser = argparse.ArgumentParser("Hyper-parameters for training model implementation")
    parser.add_argument("--feature_size", type=int, default=50, help="feature size")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="number of batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--memory_fraction", type=float, default=0.32, help="per_process_gpu_memory_fraction")
    parser.add_argument("--idx", type=int, default=0.6, help="data split %")
    parser.add_argument("--cost_rate", type=int, default=0.001425, help="cost rate")
    return parser.parse_args(args=[])
def get_input_data(code):
    yf.pdr_override()
    data_frame = pdr.get_data_yahoo(code)
    close_data = data_frame.Close
    print(close_data)
    z = [close_data[i + 1] - close_data[i] for i in range(len(close_data) - 1)]
    p = close_data[1:]
    index_bottom, index_top, f, p_t = 0, arglist.feature_size, [], []
    while index_top + arglist.feature_size <= len(z) - 1:
        f.append(z[index_bottom: index_top])
        p_t.append(p[index_bottom: index_top])
        index_top += 1
        index_bottom += 1
    z = np.asarray(z, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    p_t = np.asarray(p_t, dtype=np.float32)
    z_tp1 = f[1:,-1]
    z_tp1 = z_tp1[:,np.newaxis]
    p_t = p_t[:-1,-1]
    p_t = p_t[:, np.newaxis]
    f = f[:-1,:]
    def data_split(data, arglist):
        idx = int(len(data) * arglist.idx)
        train_data, test_data = data[:idx], data[idx:]
        return train_data, test_data
    train_f, test_f = data_split(f, arglist=arglist)
    train_z_tp1, test_z_tp1 = data_split(z_tp1, arglist=arglist)
    train_p_t, test_p_t = data_split(p_t, arglist=arglist)
    return train_f, test_f, train_z_tp1, test_z_tp1, z, train_p_t, test_p_t
def batch_gen(x,k, arglist):
    data = np.asarray(x[k:k+arglist.batch_size])
    return data
def r(delta, z_tp1, p_t, delta_tm1):
    profit_per = (delta * z_tp1 - arglist.cost_rate * p_t * abs(
            delta - delta_tm1)) / p_t
    return profit_per*100
def plt_save(y1, y2, xl, yl, ln1, ln2, name):
    plt.figure()
    plt.title(name,y=1.05)
    plt.yscale("linear")
    plt.xlabel(xl)
    plt.ylabel(yl)
    l1, = plt.plot(y1, color='blue',linewidth=0.5)
    l2, = plt.plot(y2, color='red',linewidth=0.5)
    plt.legend(handles=[l1, l2], labels=[ln1, ln2], loc='best')
    plt.savefig(name)
    plt.close()
    # plt.show()
class model():
    def __init__(self,z,  arglist):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=arglist.memory_fraction)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.Session(config=session_conf)
        self.delta_tm1 = tf.placeholder(dtype=tf.float32, shape=[arglist.batch_size, 1], name="delta_tm1")
        self.f = tf.placeholder(dtype=tf.float32, shape=[arglist.batch_size, 50], name="FuzzyIn")
        self.z = z
        self.z_tp1 = tf.placeholder(dtype=tf.float32, shape=[arglist.batch_size, 1], name="z_tp1")
        self.p_t = tf.placeholder(dtype=tf.float32, shape=[arglist.batch_size, 1], name="p_t")
        self.profit = None
        self.delta = None
        self.loss = None
        with tf.variable_scope("k-means"):
            # setting for k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness, label_kmeans, centers = cv2.kmeans(
                data=self.z, K=3, bestLabels=None, criteria=criteria, attempts=10, flags=flags)
            List0 = []
            List1 = []
            List2 = []
            for ii in range(0, self.z.size):
                if 0 == label_kmeans[ii][0]:
                    List0.append(self.z[ii])
                if 1 == label_kmeans[ii][0]:
                    List1.append(self.z[ii])
                if 2 == label_kmeans[ii][0]:
                    List2.append(self.z[ii])
            data0_tensor = tf.convert_to_tensor(value=List0, dtype=tf.float32)
            data1_tensor = tf.convert_to_tensor(value=List1, dtype=tf.float32)
            data2_tensor = tf.convert_to_tensor(value=List2, dtype=tf.float32)
            mean0, variance0 = tf.nn.moments(x=data0_tensor, axes=[0])
            mean1, variance1 = tf.nn.moments(x=data1_tensor, axes=[0])
            mean2, variance2 = tf.nn.moments(x=data2_tensor, axes=[0])
        with tf.variable_scope("fuzzy-layer"):
            fuzzy0 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.f, mean=mean0,
                                                                  variance=variance0, offset=None, scale=None,
                                                                  variance_epsilon=0.001)))
            fuzzy1 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.f, mean=mean1,
                                                                  variance=variance1, offset=None, scale=None,
                                                                  variance_epsilon=0.001)))
            fuzzy2 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.f, mean=mean2,
                                                                  variance=variance2, offset=None, scale=None,
                                                                  variance_epsilon=0.001)))
            fuzzyOut = tf.concat(values=[fuzzy0, fuzzy1, fuzzy2], axis=0, name="FuzzyOut")
            fuzzyOut = tf.reshape(tensor=fuzzyOut, shape=[arglist.batch_size, 1, 150])

        with tf.variable_scope("MLP-layer"):
            dense1 = tf.layers.dense(inputs=fuzzyOut, units=128, activation=tf.nn.sigmoid)
            dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.sigmoid)
            dense3 = tf.layers.dense(inputs=dense2, units=20, activation=tf.nn.sigmoid)
            # decoder1 = tf.layers.dense(inputs=dense4, units=50, activation=tf.nn.sigmoid)
            # decoder2 = tf.layers.dense(inputs=decoder1, units=100, activation=tf.nn.sigmoid)
            # decoder3 = tf.layers.dense(inputs=decoder2, units=150, activation=tf.nn.sigmoid)
            # loss3 = tf.losses.mean_squared_error(labels=dense3, predictions=decoder1)
            # loss2 = tf.losses.mean_squared_error(labels=dense2, predictions=decoder2)
            # loss1 = tf.losses.mean_squared_error(labels=dense1, predictions=decoder3)
            # self.pretrain_loss = (loss1+loss2+loss3)
            # train_Autoencoder = tf.train.AdamOptimizer(0.002).minimize(loss1 + loss2 + loss3)
            # optimizer = tf.train.AdamOptimizer(0.002)
            # train_op = optimizer.minimize(self.pretrain_loss)

        with tf.variable_scope("RNN-layer"):
            # vanilla_rnn_layer
            rnn_In = tf.reshape(tensor=dense3, shape=[1, arglist.batch_size, 20], name="reshape1")
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=1, activation=tf.tanh)
            initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.delta, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_In, initial_state=initial_state,
                                                        dtype=tf.float32, \
                                                        time_major=False)
            self.delta = tf.reshape(tensor=self.delta, shape=[arglist.batch_size, 1])
        with tf.variable_scope("create-loss"):
            self.profit = self.delta * self.z_tp1 - arglist.cost_rate * self.p_t* tf.abs(
                self.delta - self.delta_tm1)
            self.loss = (-1) * tf.reduce_sum(self.profit)
        with tf.variable_scope("create-profit_per"):
            self.profit_per = (self.delta * self.z_tp1 - arglist.cost_rate * self.p_t* tf.abs(
                self.delta - self.delta_tm1))/self.p_t
        with tf.variable_scope("train_p"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=arglist.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            self.timestamp = str(int(time.time()))
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.timestamp))
            print("Writing to {}\n".format(self.out_dir))
            self.loss_summary = tf.summary.scalar("loss", self.loss)

            # Train Summaries
            self.train_summary_op = tf.summary.merge([self.loss_summary])
            self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)

            # Test summaries
            self.test_summary_op = tf.summary.merge([self.loss_summary])
            self.test_summary_dir = os.path.join(self.out_dir, "summaries", "test")
            self.test_summary_writer = tf.summary.FileWriter(self.test_summary_dir, self.sess.graph)

            # Checkpoint directory. TensorFlow assumes this directory already exists so we need to create it
            self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(max_to_keep=30)
    def get_delta(self, p_t, z_tp1, delta_tm1, f):
        feed_dict = {self.p_t:p_t, self.z_tp1:z_tp1, self.delta_tm1:delta_tm1, self.f:f}
        delta = self.sess.run([self.delta],
                            feed_dict=feed_dict)
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        delta = np.asarray(delta)
        print(delta)
        delta = delta.reshape((arglist.batch_size, 1))
        for i in range(len(delta)):
            if delta[i,0]>0.3:
                delta[i,0] = 1
            elif delta[i,0]<-0.3:
                delta[i,0] = -1
            else:
                delta[i,0] = 0
        return delta
    def get_loss(self, p_t, z_tp1, delta_tm1, f):
        feed_dict = {self.p_t:p_t, self.z_tp1:z_tp1, self.delta_tm1:delta_tm1, self.f:f}
        loss = self.sess.run([self.loss],
                            feed_dict=feed_dict)
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        loss = np.asarray(loss)
        return loss
    def get_profit_per(self, p_t, z_tp1, delta_tm1, f):
        feed_dict = {self.p_t:p_t, self.z_tp1:z_tp1, self.delta_tm1:delta_tm1, self.f:f}
        profit_per = self.sess.run([self.profit_per],
                            feed_dict=feed_dict)
        # print('q_acts', q_acts.tolist(),'num',num.tolist())
        profit_per = np.asarray(profit_per)
        return profit_per*100
    def train_step(self, p_t, f, z_tp1, delta_tm1):
        feed_dict = {self.p_t:p_t, self.z_tp1:z_tp1, self.f: f, self.delta_tm1: delta_tm1}
        _, step, loss, result = self.sess.run([self.train_op, self.global_step, self.loss, self.delta], \
                                         feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
    def test_step(self,p_t, f, z_tp1, delta_tm1, writer=None):
        feed_dict = {self.p_t:p_t, self.z_tp1:z_tp1, self.f: f, self.delta_tm1: delta_tm1}
        step, summaries, loss, result, result_return = self.sess.run(
            [self.global_step, self.test_summary_op, self.loss, self.delta, self.result_return], \
            feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
        if writer:
            writer.add_summary(summaries, step)
        return result_return,
if __name__ == '__main__':
    arglist = parse_args()
    stock_list = ["000300.SS", "0050.TW", "^GSPC"]
    #################################################################
    for stock in stock_list:
        train_f, test_f, train_z_tp1, test_z_tp1, z, train_p_t, test_p_t = get_input_data(stock)
        if stock == "000300.SS":
            model = model(arglist=arglist, z=z)
            name = "SS300"
        elif stock == "0050.TW":
            name = "0050"
        else:
            name = stock
        model.saver.restore(model.sess, "FDRNN_init.ckpt")
        model.sess.run(tf.global_variables())
        # model.sess.run(tf.global_variables_initializer())
        # save_path = model.saver.save(model.sess, "FDRNN_init.ckpt")

        loss_all = []
        for epoch in range(arglist.n_epochs):
            loss_each_epoch = 0
            delta_tm1 = np.zeros((arglist.batch_size, 1))
            delta_tm1 = np.asarray(delta_tm1, dtype=np.float32)
            for k in range(len(train_f)-arglist.batch_size):
                p_t, f, z_tp1 = batch_gen(train_p_t, k, arglist), batch_gen(train_f, k, arglist), batch_gen(train_z_tp1, k, arglist)
                model.train_step(p_t=p_t, f=f, z_tp1=z_tp1, delta_tm1=delta_tm1)
                loss=  model.get_loss(p_t=p_t, f=f, z_tp1=z_tp1, delta_tm1=delta_tm1)
                loss_each_epoch += loss
                delta_tm1 = model.get_delta(p_t=p_t, f=f, z_tp1=z_tp1, delta_tm1=delta_tm1)
                current_step = tf.train.global_step(model.sess, model.global_step)
                if k == len(train_f)-arglist.batch_size-1:
                    loss_all.append(loss_each_epoch)
        plt_per(per=loss_all, name="FDRNN_train_loss_" + name , xl="epochs",
                yl="loss")

        ##############################################################
        accumulated_profit_t, accumulated_profit_t_all = 0, []
        market_t, market_all = 0,[]
        delta_list = []
        delta_tm1 = np.zeros((arglist.batch_size, 1))
        delta_tm1 = np.asarray(delta_tm1, dtype=np.float32)
        for k in range(len(test_f) - arglist.batch_size):
            p_t, f, z_tp1 = batch_gen(test_p_t, k, arglist), batch_gen(test_f, k, arglist), batch_gen(test_z_tp1, k,arglist)
            model.train_step(p_t=p_t, f=f, z_tp1=z_tp1, delta_tm1=delta_tm1)
            delta_t = model.get_delta(p_t=p_t, f=f, z_tp1=z_tp1, delta_tm1=delta_tm1)
            delta_list.append(delta_t[-1][-1])
            profit_per = r(delta_t, z_tp1, p_t, delta_tm1)
            delta_tm1 = delta_t
            r_1 = r(np.ones((z_tp1.shape)), z_tp1, p_t, np.ones((z_tp1.shape)))
            accumulated_profit_t += profit_per[-1][-1]
            market_t += np.asarray(r_1)[-1][-1]
            accumulated_profit_t_all.append(accumulated_profit_t.copy())
            market_all.append(market_t.copy())
            current_step = tf.train.global_step(model.sess, model.global_step)
        plt_per(per = accumulated_profit_t_all, name= "FDRNN_" + name + "_accumulated profit(%)", xl = "time", yl = "accumulated profit(%)")
        plt_num_act(delta_list, "time", "delta", name+"_trading_strategy")

        plt_save(accumulated_profit_t_all, market_all, "time", "accumulated profit(%)","model", "market", name)


