import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close()
########## system setting ##########
class Configure:
    batch_size = 10
    train_batch= 200
    test_batch = 192
    epochs = 150
    cost_rate = 0.0001 #0.001 ##########################
    # pretrain_learning_rate = 0.001
    learning_rate = 0.0001
    path = "D:\pythonproject\FDRNN\data\EURUSD.csv"
conf = Configure()

########## data processing ##########
def read_data(path):
    df = pd.read_csv(path, header=0, index_col=0)
    price = df["Close"].tolist()
    return price
def data_processing():
    price=read_data(conf.path)
    # print(len(price),type(price))
    # plt.plot(price)
    # plt.show()
    ### 製造價差data
    zt=[price[i]-price[i-1] for i in range(1,len(price))] #4918list
    momentum=[50,60,80,120,240]
    for i in momentum:
        globals()['zt%s' % i]=[price[240+j]-price[240-i+j] for j in range(0,len(price)-240)] #4679list
    pct=[(zt[i]/price[i])*100 for i in range(len(zt))] #4918list
    ft=[zt[i:i+45][::-1] for i in range(195,(len(zt)-44))] #4679,45list
    # print(ft, len(ft), len(ft[0]),sep='\n')
    ### ft45->50
    for i in range(len(ft)):
        ft[i].extend([zt50[i],zt60[i],zt80[i],zt120[i],zt240[i]]) #4679,50list
    # print(ft[0], len(ft), len(ft[0]),sep='\n')
    ### cost
    c = [price[i] * conf.cost_rate for i in range(239, (len(zt)))]  # 4679-234=4445list
    # print(c,type(c),len(c),sep='\n')
    ###　label
    label = zt[239:(len(zt))]
    del_index = [i * conf.batch_size for i in range(len(ft) // conf.batch_size + 1)]  # [0,20,...,4640] 234
    # print(del_index, len(del_index), sep='\n')
    c = [c[j] for j in range(len(c)) if (j not in del_index)]  # 4679-234=4445list
    label = [label[j] for j in range(len(label)) if (j not in del_index)]  # 4679-234=4445list
    # print(label,len(label),sep='\n')
    ## 將資料行式轉為可輸入格式
    zt = np.asarray(zt, dtype=np.float32).reshape(-1, 1)
    pct = np.asarray(pct, dtype=np.float32).reshape(-1, 1)
    ft = np.asarray(ft, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32).reshape(-1, 1)
    label = np.asarray(label, dtype=np.float32).reshape(-1, 1)
    return zt, pct, ft, c, label
zt, pct, ft, c, label=data_processing()
# print(zt, pct, ft, c, label,sep='\n\n')


############ model ##########
from sklearn.cluster import KMeans
import tensorflow.compat.v1 as tf
tf.disable_eager_execution() #將tf.2.降版本，為讀1.版的語法
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class Model:
    def __init__(self,zt): #激活object即有init
        self.Input_Data = tf.placeholder(dtype=tf.float32, shape=[conf.batch_size, 50], name="Input_Data") #tf.placeholder讀入資料行式
        self.Label = tf.placeholder(dtype=tf.float32, shape=[conf.batch_size-1, 1], name="Label")
        self.Cost = tf.placeholder(dtype=tf.float32, shape=[conf.batch_size-1, 1], name="Cost")
        self.delta = None
        self.loss = None
        self.result_return = None

        with tf.variable_scope("k_means"):
            clusters = KMeans(n_clusters=3,n_init=150, max_iter=100, tol=1e-7, random_state=1,n_jobs=-1).fit(zt)
            for i in range(3):
                globals()['clst%s' % i] = [zt[j][0] for j, k in enumerate(clusters.labels_) if k == i]
            # print(clusters.labels_,clusters.cluster_centers_,sep='\n')
            # print(clst0,clst1,clst2,sep='\n') #平盤 多 空
            # print(len(clst0),len(clst1),len(clst2),len(clst0)+len(clst1)+len(clst2),sep=' ') #4918
            clst_ts0 = tf.convert_to_tensor(value=clst0, dtype=tf.float32)  # tf.convert_to_tensor
            clst_ts1 = tf.convert_to_tensor(value=clst1, dtype=tf.float32)
            clst_ts2 = tf.convert_to_tensor(value=clst2, dtype=tf.float32)
            # with tf.Session() as sess:
            #     print(sess.run(data0_tensor))
            mean0, variance0 = tf.nn.moments(x=clst_ts0, axes=[0])
            mean1, variance1 = tf.nn.moments(x=clst_ts1, axes=[0])
            mean2, variance2 = tf.nn.moments(x=clst_ts2, axes=[0])
            # with tf.Session() as sess:
            #    print(sess.run(mean1),sess.run(variance1))

        with tf.variable_scope("fuzzy_layer"):
            fuzzy0 = tf.exp(-(tf.nn.batch_normalization(x=self.Input_Data, mean=mean0,
                                                                  variance=variance0, offset=None, scale=None,
                                                                  variance_epsilon=1e-5))**2)
            fuzzy1 = tf.exp(-(tf.nn.batch_normalization(x=self.Input_Data, mean=mean1,
                                                                  variance=variance1, offset=None, scale=None,
                                                                  variance_epsilon=1e-5))**2)
            fuzzy2 = tf.exp(-(tf.nn.batch_normalization(x=self.Input_Data, mean=mean2,
                                                                  variance=variance2, offset=None, scale=None,
                                                                  variance_epsilon=1e-5))**2)
            fuzzy = tf.concat(values=[fuzzy0, fuzzy1, fuzzy2], axis=0, name="fuzzy") #tf.concat()等同於extend在後面
            fuzzy_rep = tf.reshape(tensor=fuzzy, shape=[conf.batch_size, 1, 150])
            # with tf.Session() as sess:
            #     fuzzy_rep = sess.run(fuzzy_rep, feed_dict={self.Input_Data: ft[0:20,:]})
            #     print(fuzzy_rep[0])

        with tf.variable_scope("DNN_layer"):
            dense1 = tf.layers.dense(inputs=fuzzy_rep, units=150, activation=tf.sigmoid)
            dense2 = tf.layers.dense(inputs=dense1, units=80, activation=tf.sigmoid)
            dense3 = tf.layers.dense(inputs=dense2, units=40, activation=tf.sigmoid)
            dense4 = tf.layers.dense(inputs=dense3, units=20)
            #dnn.summary()
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())  # when it's end,initialize all variables
            #     dense4 = sess.run(dense4, feed_dict={self.Input_Data: ft[0:20,:]})
            #     print(dense4,dense4.shape,sep='\n')

        with tf.variable_scope("RNN_layer"):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=1, activation=tf.tanh)
            Ft = tf.reshape(tensor=dense4, shape=[conf.batch_size, 1, 20], name="reshape1") #[max_time, batch_size, cell_state_size]
            initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.delta, final_state = tf.nn.dynamic_rnn(rnn_cell, Ft, initial_state=initial_state,
                                                        dtype=tf.float32,time_major=True)
            self.delta = tf.reshape(tensor=self.delta, shape=[conf.batch_size, 1])
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     self.delta,self.decision = sess.run([self.delta,self.decision],feed_dict={self.Input_Data: ft[0:20,:]})
            #     print(self.delta,self.decision,sep='\n')

        with tf.variable_scope("loss_function"):
            self.rt = [self.delta[i:i+1][0][0] * self.Label[i:i+1][0][0] - self.Cost[i:i+1][0][0] * \
                                  tf.abs(self.delta[i + 1:i+2][0][0] - self.delta[i:i+1][0][0]) for i in range(conf.batch_size-1)]
            self.loss = (-1) * tf.reduce_sum(self.rt)
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     result_return,loss = sess.run([self.result_return,self.loss], feed_dict={self.Input_Data: ft[0:20, :],
            #                                                        self.Label: label[0:19,:],
            #                                                        self.Cost: c[0:19,:]})
            #     print(result_return,len(result_return),loss,sep='\n')

model0=Model(zt)

##########Split data and label set##########
train_data, train_label, train_cost = ft[:conf.train_batch*conf.batch_size],\
                                      label[:conf.train_batch*(conf.batch_size-1)],\
                                      c[:conf.train_batch*(conf.batch_size-1)]
test_data, test_label, test_cost = ft[conf.train_batch*conf.batch_size:],\
                                   label[conf.train_batch*(conf.batch_size-1):],\
                                   c[conf.train_batch*(conf.batch_size-1):]

############ traning ############
import datetime

with tf.Graph().as_default():
    ### 設定使用資源
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45) #GPU使用率50%
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model=Model(zt) #實體化class
        global_step = tf.Variable(0, name="iter", trainable=False) #計算迭代次數
        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
        train_op = optimizer.minimize(model.loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def model_training(data_batch, label_batch, cost_batch):
            train, iter, decision, loss = sess.run([train_op, global_step, model.delta, model.loss],
                                                   feed_dict={model.Input_Data: data_batch,
                                                              model.Label: label_batch,
                                                              model.Cost:cost_batch})
            # time_str = datetime.datetime.now().isoformat()
            # print("iter {}, loss {:g}".format(iter, loss))
            # print('decision=',decision.reshape(1, -1)[0])
            return decision, loss

        def model_testing(data_batch, label_batch, cost_batch):
            _,iter, decision, loss= sess.run([train_op,global_step, model.delta, model.loss],
                                                         feed_dict={model.Input_Data: data_batch,
                                                                    model.Label: label_batch,
                                                                    model.Cost:cost_batch})
            # print("iter {}, loss {:g}".format(iter, loss))
            # print('decision=', decision.reshape(1, -1)[0])
            return decision, loss

        # price = read_data(conf.path)
        # year = abs(4165 - (len(price) // 20) * 20) / 120
        # air = 0.1
        # least_return = (1 + air) ** year*price[4615]-price[4615]

        saver = tf.train.Saver()
        for jj in range(10,11): #train jj 個model出來
            epoch_loss = [1] * conf.epochs
            ii=0
            while epoch_loss[conf.epochs-1]>=0:
                sess.run(tf.global_variables_initializer())  # when it's end,initialize all variables
                for epoch in range(conf.epochs):
                    total_loss=0
                    for i in range(conf.train_batch):
                        x_batch, y_batch, c_batch = train_data[i * conf.batch_size:(i + 1) * conf.batch_size],\
                                                train_label[i * (conf.batch_size-1):(i + 1) * (conf.batch_size-1)],\
                                                train_cost[i * (conf.batch_size-1):(i + 1) * (conf.batch_size-1)]
                        decision, loss=model_training(x_batch, y_batch, c_batch)
                        current_step = tf.train.global_step(sess, global_step)
                        total_loss+=loss
                    epoch_loss[epoch]=total_loss
                ii+=1
                print(ii,'loop:','loss=',epoch_loss[-1])
            else:
                path = './eurmodel{}.ckpt'.format(jj)
                save_path = saver.save(sess, path)
                print("========== training ==========")
                print(epoch_loss[-1])
                plt.title("Epoch loss")
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(epoch_loss)
                plt.savefig("eurmodel{}_loss.png".format(jj), dpi=300, bbox_inches='tight')
                plt.show()

############ testing ############
                decision_ls = []
                ac_return, return_ls = 0, []
                saver.restore(sess, path)
                for i in range(conf.test_batch):
                    x_batch, y_batch, c_batch = test_data[i * conf.batch_size:(i + 1) * conf.batch_size], \
                                                test_label[i * (conf.batch_size-1):(i + 1) * (conf.batch_size-1)], \
                                                test_cost[i * (conf.batch_size-1):(i + 1) * (conf.batch_size-1)]
                    decision, loss = model_testing(x_batch, y_batch, c_batch)
                    decision_ls.extend(decision.reshape(1,-1)[0])
                    ac_return-=loss
                    return_ls.append(ac_return)
                print("========== testing ==========")
                print(decision_ls)
                day=[i+1 for i in range(len(decision_ls))]
                plt.title("Decision")
                plt.xlabel('10days')
                plt.ylabel('decision')
                plt.scatter(day,decision_ls,s =2)
                plt.legend(handles=[1, 0,-1],labels=['long','empty','short'], loc='best')
                plt.savefig("eurmodel{}_decision.png".format(jj), dpi=300, bbox_inches='tight')
                plt.show()
                print(return_ls[-1])
                plt.title("Accumulation_return")
                plt.xlabel('10days')
                plt.ylabel('return')
                plt.plot(return_ls)
                plt.savefig("eurmodel{}_Accumulation_return.png".format(jj), dpi=300, bbox_inches='tight')
                plt.show()
