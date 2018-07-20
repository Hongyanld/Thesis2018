"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from main_2_test import word_embedded

BATCH_START = 0
TIME_STEPS = 50
BATCH_SIZE = 25
INPUT_SIZE = 300
OUTPUT_SIZE = 11
CELL_SIZE = 512
LR = 0.006






def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (25batch, 50steps)
    test_x = word_embedded
    test_y = mnist.test.labels[:2000]
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

def add_input_layer():
    l_in_x = tf.reshape(xs, [-1, input_size], name='2_2D')  # (batch, n_step, in_size).>>>>(batch*n_step, in_size)
    # Ws (in_size, cell_size)
    Ws_in = _weight_variable([input_size, cell_size])
    # bs (cell_size, )
    bs_in = _bias_variable([cell_size, ])
    # l_in_y = (batch * n_steps, cell_size)
    with tf.name_scope('Wx_plus_b'):
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
    l_in_y = tf.reshape(l_in_y, [-1, n_steps, cell_size], name='2_3D')



def add_cell(  ):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)
    with tf.name_scope('initial_state'):
        cell_init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
        lstm_cell, l_in_y, initial_state=cell_init_state, time_major=False)


def add_output_layer(  ):
    # shape = (batch * steps, cell_size)
    l_out_x = tf.reshape(cell_outputs, [-1, cell_size], name='2_2D')
    Ws_out = _weight_variable([cell_size, output_size])
    bs_out = _bias_variable([output_size, ])
    # shape = (batch * steps, output_size)
    with tf.name_scope('Wx_plus_b'):
        pred = tf.matmul(l_out_x, Ws_out) + bs_out





    def __init__(  , n_steps, input_size, output_size, cell_size, batch_size):
         n_steps = n_steps      #50
         input_size = input_size #300
         output_size = output_size  #128
         cell_size = cell_size
         batch_size = batch_size
        with tf.name_scope('inputs'):
             xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
             ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
             add_input_layer()
        with tf.variable_scope('LSTM_cell'):
             add_cell()
        with tf.variable_scope('out_hidden'):
             add_output_layer() #在哪个define里面
        # with tf.name_scope('cost'):
        #      compute_cost()
        with tf.name_scope('train'):
             train_op = tf.train.AdamOptimizer(LR).minimize( cost)
        with tf.name_scope('correct_pred'):
             ()

    def add_input_layer(  ,):
        l_in_x = tf.reshape( xs, [-1,  input_size], name='2_2D')  # (batch, n_step, in_size).>>>>(batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in =  _weight_variable([ input_size,  cell_size])
        # bs (cell_size, )
        bs_in =  _bias_variable([ cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
         l_in_y = tf.reshape(l_in_y, [-1,  n_steps,  cell_size], name='2_3D')


    def add_cell(  ):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell( cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
             cell_init_state = lstm_cell.zero_state( batch_size, dtype=tf.float32)
         cell_outputs,  cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell,  l_in_y, initial_state= cell_init_state, time_major=False)-

    def add_output_layer(  ):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape( cell_outputs, [-1,  cell_size], name='2_2D')
        Ws_out =  _weight_variable([ cell_size,  output_size])
        bs_out =  _bias_variable([ output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
             pred = tf.matmul(l_out_x, Ws_out) + bs_out


    def compute_cost(  ):
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= pred, labels= ys)
        tf.summary.scalar('cost',  cost)

    def compute_correct_pred(  ):
         correct_pred() = tf.equal(tf.argmax( pred, 1), tf.argmax( ys, 1))

    def compute_accuracy(  ):
         accuracy = tf.reduce_mean(tf.cast( correct_pred(), tf.float32))


    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(  , shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(  , shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'  #通过log 之后，直接用Google charm 0.0.0.0:6006 得到已经设计好的整个框架。

    plt.ion()
    plt.show()
    for i in range(200):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
