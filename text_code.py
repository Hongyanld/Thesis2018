# import numpy as np
import tensorflow as tf
# from keras import Input, Model
# from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTM, Lambda, Concatenate, \
#     MaxPooling1D, Conv1D, GlobalMaxPool1D, concatenate, GRU
# from keras.optimizers import RMSprop
#
# from main import word_embeddings_size, args, max_num_of_sentences, \
#     max_num_of_tokens_per_sentence

# a = np.zeros((1, 3))
# b = np.ones((1, 3))
# a[0:1] = b
#
#
# data = [[1,2,3],[4,5,6]]
# data_np = np.asarray(data, np.float32)
#
# data_tf = tf.convert_to_tensor(data_np, np.float32)
# sess = tf.InteractiveSession()
# print(data_tf.eval())
#
# sess.close()
#
#
# tup = ('physics', 'chemistry', 1997, 2000);
# x = [(1,2), (3,4), (5,6)]
#
# simple example
#
# import tensorflow as tf
#  #定义‘符号’变量，也称为占位符
#  a = tf.placeholder("float")
#  b = tf.placeholder("float")
#
#  y = tf.mul(a, b) #构造一个op节点
#
#  sess = tf.Session()#建立会话
#  #运行会话，输入数据，并计算节点，同时打印结果
#  print sess.run(y, feed_dict={a: 3, b: 3})
#  # 任务完成, 关闭会话.
#  sess.close()
#
# #探索三种方式定义的变量之间的区别
# # 1.placeholder
# v1 = tf.placeholder(tf.float32, shape=[2,3,4], name='ph')
# print v1.name
# v1 = tf.placeholder(tf.float32, shape=[2,3,4], name='ph')
# print v1.name
#
# # 2. tf.Variable()
# v2 = tf.Variable([1,2], dtype=tf.float32, name='V')
# print v2.name
# v2 = tf.Variable([1,2], dtype=tf.float32, name='V')
#
# # 3.tf.get_variable() 创建变量的时候必须要提供 name
# v3 = tf.get_variable(name='gv', shape=[])
# print v3.name
# v4 = tf.get_variable(name='gv', shape=[2])
# print v4.name
# # 第三种当命名一样的时候结果会发生冲突。
#
# #返回所有可以 train的变量
# vs = tf.trainable_variables()
# print len(vs)
# for v in vs:
#     print v
#
# 4
# Tensor("Variable/read:0", shape=(2,), dtype=float32)
# Tensor("V/read:0", shape=(2,), dtype=float32)
# Tensor("V_1/read:0", shape=(2,), dtype=float32)
# Tensor("gv/read:0", shape=(), dtype=float32)
# 结论：第一种变量不能train
#
# # 三种方式所定义的变量
# tf.placeholder() 占位符。* trainable==False *
# tf.Variable() 一般变量用这种方式定义。 * 可以选择 trainable 类型 *
# tf.get_variable() 一般都是和 tf.variable_scope() 配合使用，从而实现变量共享的功能。 * 可以选择 trainable 类型 *
#
# 探索 name_scope 和 variable_scope：
# with tf.name_scope('nsc1'):
#     v1 = tf.Variable([1], name='v1')
#     v1 = tf.Variable([1], name='v1')
#     with tf.variable_scope('vsc1'):
#         v2 = tf.Variable([1], name='v2')
#         v3 = tf.get_variable(name='v3', shape=[])
# print 'v1.name: ', v1.name
# print 'v2.name: ', v2.name
# print 'v3.name: ', v3.name

# def _weight_variable(shape, name='weights'):
#     initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
#     weight = tf.get_variable(shape=shape, initializer=initializer, name=name)
#     #weight = tf.Variable(shape, initializer, name=name)
#     return weight
#
# def _bias_variable(shape, name='biases'):
#     initializer = tf.constant_initializer(0.1)
#     bias = tf.get_variable(name=name, shape=shape, initializer=initializer)
#     #bias = tf.Variable(name=name, shape=shape, initializer=initializer)
#     return bias
#     LR = 0.006
#     n_steps = 50  # 50
#     input_size = 300  # 300
#     output_size = 256  # 128
#     cell_size = 512
#     batch_size = 25
#
#
#     n_steps_2 = 25
#     input_size_2 = 256
#     output_size_2 = 11  # 128
#     cell_size_2 = 256
#     batch_size_2 = 25
#     batch_start = 0
#     with tf.variable_scope('LSTM_cell'):
#         Ws_out_2 = _weight_variable([cell_size_2, output_size_2],)
#
#     with tf.variable_scope('out_hidden'):
#         Ws_in = _weight_variable([input_size, cell_size])




import numpy as np

# A = [[1, 3, 4, 5, 6]]
# B = [[1, 3, 4, 3, 2]]
#
# with tf.Session() as sess:
#     print(sess.run(tf.equal(A, B)))
#
# cross_entropy loss function
# #batch_size = 2
# labels = tf.constant([[0, 0, 0, 1],[0, 1, 0, 0]])
# logits = tf.constant([[-3.4, 2.5, -1.2, 5.5],[-3.4, 2.5, -1.2, 5.5]])
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# with tf.Session() as sess:
#     print("softmax loss:", sess.run(loss))
#
#
# a=tf.argmax(labels,1)
# with tf.Session() as sess:
#     print("labels", sess.run(a))
#
#
# loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1), logits=logits)
# with tf.Session() as sess:
#     print ("sparse softmax loss:", sess.run(loss_s))
# cost = tf.reduce_mean(loss_s)

# tensor to numpy
# import tensorflow as tf
# img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
# img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
# img = tf.concat(values=[img1,img2],axis=3)
# sess=tf.Session()
# #sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
# print("out1=",type(img))
# #转化为numpy数组
# img_numpy=img.eval(session=sess)
# print("out2=",type(img_numpy))
# #转化为tensor
# img_tensor= tf.convert_to_tensor(img_numpy)
# print("out2=",type(img_tensor))

# def _weight_variable(shape, name='weights'):
#     initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
#     weight = tf.get_variable(shape=shape, initializer=initializer, name=name)
#     return weight
#
#
# with tf.variable_scope('out_hidden'):
#     Ws_out = _weight_variable([4, 4])
#     print('Ws_out',Ws_out)
#
# with tf.variable_scope('in_hidden'):
#     Ws_out_2 = _weight_variable([4, 4])
#     print('Ws_out_2', Ws_out_2)
#
# Ws_out_3 = Ws_out_2 + Ws_out_2
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# sess.run(Ws_out)
# # sess.run(Ws_out_2)
# for i in range(2):
#     i += 2
#     print('i',i)
#
# i=2
# for j in range(3):
#     j += i
#     print('j',j)



#
# for i in range(10):
#     a = np.array([0,1,2,3,4,5])
#     b = a[0]
#     print (i)
#     i += 2

# import matplotlib.pyplot as plt
#
# x = [1, 2, 3, 4, 5]
# y = [1, 4, 9, 16, 25]
#
# plt.plot(x, y)
# plt.show()
import numpy as np
import matplotlib.pyplot as pl

x1 = [1, 2, 3, 4, 5]  # Make x, y arrays for each graph
y1 = [1, 4, 9, 16, 25]
x2 = [1, 2, 4, 6, 8]
y2 = [2, 4, 8, 12, 16]

pl.plot(x1, y1, 'r')# use pylab to plot x and y
pl.plot(x2, y2, 'g')
pl.show()


#

# pl.title(’Plot
# of
# y
# vs.x’)  # give plot a title
# pl.xlabel(’x
# axis’)  # make axis labels
# pl.ylabel(’y
# axis’)
#
#
# pl.xlim(0.0, 9.0)  # set axis limits
# pl.ylim(0.0, 30.)
