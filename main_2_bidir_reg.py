#! /usr/bin/env python3
# Copyright (C) 2018 Robert Werfelmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import pickle
import sys
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import cpu_count
from re import sub
from string import punctuation
from time import time

from enchant.checker import SpellChecker
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from keras import callbacks
from keras.utils import to_categorical
from nltk import pos_tag
from pandas import DataFrame, concat, read_table
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

import keras_resnet
import keras_resnet.blocks
import keras_resnet.models
import numpy as np
import tensorflow as tf
from keras import Input, Model,regularizers
from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTMCell, Lambda, Concatenate, \
    MaxPooling1D, Conv1D, GlobalMaxPool1D, concatenate, GRU, GlobalAveragePooling1D
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt




from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument('-m', '--model', action='store_true')
parser.add_argument('-f', '--feature_matrix', action='store_true')
parser.add_argument('-ds', '--doc_sequence_embedded', action='store_true')
parser.add_argument('-w', '--word_vector', action='store_true')
parser.add_argument('-k', '--k_value', metavar='INT', type=int, default=5)
parser.add_argument('-d', '--data_file', metavar='PATH', type=str, default='toefl11_tokenized.tsv')
parser.add_argument('-d2', '--data_file_test', metavar='PATH', type=str, default='toefl11_tokenized_test.tsv')

parser.add_argument('-s', '--sequence', metavar='TYPE', type=str, default='word',
                    help='TYPE can be one of the following: "pos", "word", "char", "combined", "combined_2')
parser.add_argument('-n', '--neural_network', metavar='TYPE', type=str, default='lstm',
                    help='TYPE can be one of the following: "lstm", "conv"')
parser.add_argument('--lstm_unit_size', metavar='INT', type=int, default=256)
parser.add_argument('--learning_rate', metavar='INT', type=float, default=0.0005)
args = parser.parse_args()

root_path = "./"
word_embeddings_size = 300
pos_embeddings_size = 128
char_embeddings_size = 100
additional_feature_list = ['num_of_tokens',
                           # 'avg_word_len',
                           'det_token_ratio',
                           'punct_token_ratio',
                           'num_of_long_sentences',
                           'num_of_short_sentences']  # 'num_of_passive_sentences']
max_num_of_sentences = 25
max_num_of_tokens_per_sentence = 50  # Covers about 98% of data
max_num_of_chars = 300


def read_data(file):
    print("Reading index from", file)
    return read_table(file)


def get_word2vec_model(text_series):
    if args.model:
        print("Loading word2vec models from", root_path + "model_2_bidir_reg/")
        word_model = Word2Vec.load(root_path + "model_2_bidir_reg/toefl11.word_model.bin")
    else:
        print("Training word2vec model")
        word_model = train_word2vec_model(text_series)

    return word_model


def train_word2vec_model(text_series):
    print("\tBuilding lists of tokens")
    word_data = [sent.lower().split() for doc in text_series for sent in doc.split('\n') if sent]

    #word_data_text[1] = [sent.lower().split() for sent in text_series[i].split('\n') if sent]

    epochs = 70
    print("\tTraining word2vec word model with", epochs, "epochs")
    start = time()
    word_model = Word2Vec(word_data, size=word_embeddings_size, alpha=0.02, window=1, min_count=1,
                          workers=cpu_count(), sg=1, iter=epochs)
    print("\tTrained word2vec word model in", (time() - start) / 60, "minutes")

    print("\tSaving word2vec models to", root_path + "model_2_bidir_reg/")
    word_model.save(root_path + "model_2_bidir_reg/toefl11.word_model.bin")

    return word_model

def train_word_vector(word_model):
    vocab = word_model.wv.vocab
    word_vector = {}
    for word in vocab:
        word_vector[word] = word_model[word]
    word_vector_add = {0: np.zeros(300)};
    word_vector.update(word_vector_add)
    print("\tSaving word_vector to", root_path + "model_2_bidir_reg/")
    with open(root_path + "model_2_bidir_reg/"  + "text_series.word_vector.pickle", "wb") as file_to_write:
        pickle.dump(word_vector, file_to_write, protocol=4)
    return word_vector

def get_word_vector(word_model):
    if args.word_vector: #?
        print("Loading word_vector from", root_path + "model_2_bidir_reg/")
        with open(root_path + "model_2_bidir_reg/"  + "text_series.word_vector.pickle", "rb") as file_to_read:
            word_vector = pickle.load(file_to_read)
    else:
        word_vector = train_word_vector(word_model)
    return word_vector


def get_feature_matrix(text_series, word_indices):
    if args.feature_matrix:
        print("Loading feature matrices from", root_path + "model_2_bidir_reg/")
        with open(root_path + "model_2_bidir_reg/" + args.data_file + ".word_sequence_matrix.pickle", "rb") as file_to_read:
            word_sequence_matrix = pickle.load(file_to_read)
    else:
        print("\tBuilding Word sequence matrix")
        word_sequence_matrix = np.stack([build_sentence_word_sequences(
            text_series[index], word_indices) for index in range(len(text_series))])
        with open(root_path + "model_2_bidir_reg/" + args.data_file + ".word_sequence_matrix.pickle", "wb") as file_to_write:
            pickle.dump(word_sequence_matrix, file_to_write, protocol=4)

    return word_sequence_matrix



def build_sentence_word_sequences(text, word_indices):
    doc_matrix = np.zeros((max_num_of_sentences, max_num_of_tokens_per_sentence), dtype=np.int64) #便捷将没有单词的设置为0
    for i, sentence in enumerate(text.split('\n')):
        if i < max_num_of_sentences:
            for j, word in enumerate(sentence.lower().split()[:max_num_of_tokens_per_sentence]):
                doc_matrix[i, j] = word_indices[word]# 这里用word 的index 来表示doc_matrix中的一个元素（这样能体现出来word embedding的意义么？
    return doc_matrix




# def
#     word_sequence_matrix = get_feature_matrix(text_series, word_indices)
#     text_word_sequence_embedded = np.stack([get_text_word_embedded(
#         text_series[index], word_indices, word_indices_inv, word_vector) for index in range(len(text_series))])


# def get_doc_sequence_embedded(text_series,word_indices,word_vector,word_indices_inv):
#     if args.doc_sequence_embedded:
#         print("Loading text_word_sequence_embedded from", root_path + "model_2_bidir_reg/")
#         with open(root_path + "model_2_bidir_reg/" + args.data_file + ".doc_sequence_embedded.pickle", "rb") as file_to_read:
#             doc_sequence_embedded = pickle.load(file_to_read)
#     else:
#         print("\tBuilding doc_sequence_embedded")
#         word_sequence_matrix = get_feature_matrix(text_series,word_indices)
#         doc_sequence_embedded = np.stack(
#             (word_vector[word_indices_inv[word_sequence_matrix[test_series[index], i, j]]]) for index in range(0, 12100)
#             for i in range(max_num_of_sentences) for j in range(max_num_of_tokens_per_sentence))
#         with open(root_path + "model_2_bidir_reg/" + args.data_file + ".doc_sequence_embedded.pickle", "wb") as file_to_write:
#             pickle.dump(text_word_sequence_embedded, file_to_write, protocol=4)
#     return doc_sequence_embedded

def get_spelling_errors(text):
    return " ".join([err.word for err in SpellChecker("en_US", text)]) or ""


# record history of training
class LossHistory(callbacks.Callback):
    def __init__(self):
        super().__init__()
        logs = {}
        accuracies = []
        losses = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        losses.append(logs.get('loss'))
        accuracies.append(logs.get('acc'))



def get_total_sentence_count(text_series):
    return sum([len(text_series[i].split('\n')) for i in range(text_series.shape[0])])


def plot_text_lengths(doc_text):
    import matplotlib.pyplot as plt
    import seaborn as sns

    list_of_pos_tags = Parallel(n_jobs=cpu_count())(delayed(get_pos_tag_sequence)(text) for text in doc_text)
    sent_lengths = [len(doc) for doc in list_of_pos_tags]
    token_lengths = [len(sent) for doc in list_of_pos_tags for sent in doc]


    plt.title("Number of sentences per document")
    plt.xlabel("Average number of sentences: " + str(np.mean(sent_lengths)) + "\nLargest number of sentences: " + str(
        max(sent_lengths)))
    plt.ylabel(str(int(len([len(doc) for doc in list_of_pos_tags if len(doc) < 25]) / len(
        list_of_pos_tags) * 100)) + "% of documents are under 25 lengths")
    plt.axis([0, 80, 0, 1])
    sns.boxplot(sent_lengths)
    plt.show()

    plt.title("Number of tokens per sentence")
    plt.xlabel("Average token length per sentence: " + str(
        np.mean(token_lengths)) + "\nLargest token length per sentence: " + str(max(token_lengths)))
    plt.ylabel(str(int(len([len(sent) for doc in list_of_pos_tags for sent in doc if len(sent) < 50]) / len(
        token_lengths) * 100)) + "% of sentences have under 50 tokens")
    plt.axis([0, 80, 0, 1])
    sns.boxplot(token_lengths)
    plt.show()


def find_word_n_grams(word_list, n=2):
    return zip(*[word_list[i:] for i in range(n)])



def train_val_test_split(df):

    x_train, x_test, y_train, y_test = train_test_split(df, df['Language'], stratify=df[['Language']], #, 'Prompt'
                                                        test_size=0.1, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, x_test['Language'], stratify=x_test[['Language']], #, 'Prompt'
                                                    test_size=1, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test


# def rnn_relu(kernel_shape, bias_shape):
#     with tf.variable_scope("Ws_bs"):
#         bs_in_1 = tf.get_variable("biases", bias_shape, initializer=initializer = tf.constant_initializer(0.1))
#         bs_out_1 = tf.get_variable("biases", bias_shape, initializer=initializer = tf.constant_initializer(0.1))
#         bs_out_2 = tf.get_variable("biases", bias_shape, initializer=initializer = tf.constant_initializer(0.1))
#         Ws_in_1 = tf.get_variable("weights", kernel_shape,
#                                    initializer=tf.random_normal_initializer(mean=0., stddev=1., ))
#         Ws_out_1 = tf.get_variable("weights", kernel_shape,
#                                    initializer=tf.random_normal_initializer(mean=0., stddev=1., ))
#         Ws_out_2= tf.get_variable("weights", kernel_shape,
#                                    initializer=tf.random_normal_initializer(mean=0., stddev=1., ))
#         return None
# with tf.name_scope("a_name_scope"):
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
#     var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
#     var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
#     var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)
def _weight_variable(shape, name='weights'):
    initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
    weight = tf.get_variable(shape=shape, initializer=initializer, trainable=True, name=name)
    return weight

def _bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    bias = tf.get_variable(name=name, shape=shape, trainable=True, initializer=initializer)
    return bias

def main():
    df = read_data(args.data_file)

    labels_series = df['Language']
    text_series = df['Text']

    unique_labels_count = len(Counter(labels_series))

    word_model= get_word2vec_model(text_series)


    word_vector = get_word_vector(word_model)

    word_indices = dict((p, i) for i, p in enumerate(word_model.wv.vocab, start=1))
    word_indices_inv = dict((p, i) for p, i in enumerate(word_model.wv.vocab, start=1))


    word_sequence_matrix = get_feature_matrix(text_series,word_indices)


    if args.doc_sequence_embedded:
        print("Loading text_word_sequence_embedded from", root_path + "model_2_bidir_reg/")
        with open(root_path + "model_2_bidir_reg/" + args.data_file + ".doc_sequence_embedded.pickle", "rb") as file_to_read:
            doc_sequence_embedded = pickle.load(file_to_read)
    else:
        print("\tBuilding doc_sequence_embedded")
        doc_sequence_embedded = np.stack(
            [word_vector.get(word_indices_inv.setdefault(word_sequence_matrix[index, i, j],0)) for index in range(0, len(text_series))
            for i in range(max_num_of_sentences) for j in range(max_num_of_tokens_per_sentence)])
        with open(root_path + "model_2_bidir_reg/" + args.data_file + ".doc_sequence_embedded.pickle", "wb") as file_to_write:
            pickle.dump(doc_sequence_embedded, file_to_write, protocol=4)


    encoder = LabelEncoder().fit(labels_series)
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    # history = LossHistory()

    x_train, x_test, y_train, y_test = train_test_split(df, df['Language'], stratify=df[['Language']], #, 'Prompt'
                                                        test_size=0.1, random_state=0)

    x_val = x_test
    y_val = y_test




    # define input and output
    one_hot_train_labels = to_categorical(encoder.transform(y_train), num_classes=unique_labels_count)
    one_hot_val_labels = to_categorical(encoder.transform(y_val), num_classes=unique_labels_count)

    # start model train:
    epoch_num = 50
    print("Training neural network for", epoch_num, "epoch(s)")
    doc_word_num = max_num_of_sentences * max_num_of_tokens_per_sentence
    word_embedded_train = np.zeros((len(x_train.index) * doc_word_num, word_embeddings_size),dtype=np.float32)
    for i in range(0,len(x_train.index)):
        word_embedding_index_start=x_train.index[i]*doc_word_num
        word_embedding_index_end=(x_train.index[i]+1)*doc_word_num
        word_embedded_train[i * doc_word_num:(i + 1) * doc_word_num] = doc_sequence_embedded[word_embedding_index_start:word_embedding_index_end]

    word_embedded_train_tf = tf.convert_to_tensor(word_embedded_train, np.float32)

    # sess = tf.InteractiveSession()
    # print(word_embedded_train_tf.eval())
    # sess.close()

    word_embedded_val = np.zeros((len(x_val.index) * doc_word_num, word_embeddings_size), dtype=np.float32)
    for i in range(0, len(x_val.index)):
        word_embedding_index_start = x_val.index[i] * doc_word_num
        word_embedding_index_end = (x_val.index[i] + 1) * doc_word_num
        word_embedded_val[i * doc_word_num:(i + 1) * doc_word_num] = doc_sequence_embedded[
                                                                       word_embedding_index_start:word_embedding_index_end]

    # word_embedded_val = tf.convert_to_tensor(word_embedded_val, np.float32)
    # sess = tf.InteractiveSession()
    # print(word_embedded_val.eval())
    # sess.close()

    # build LSTM network
    # BATCH_START = 0
    # TIME_STEPS = 50
    # BATCH_SIZE = 25
    # INPUT_SIZE = 300
    # OUTPUT_SIZE = 256
    # CELL_SIZE = 512
    # TIME_STEPS_2 = 25
    # BATCH_SIZE_2 = 25
    # INPUT_SIZE_2 = 256
    # OUTPUT_SIZE_2 = 11
    # CELL_SIZE_2 = 128

    LR = 0.006
    n_steps = 50  # 50
    input_size = 300  # 300
    output_size = 256  # 128
    cell_size = 512
    batch_size = 100


    n_steps_2 = 25
    input_size_2 = 256
    output_size_2 = 11  # 128
    cell_size_2 = 256
    batch_size_2 = 4
    batch_start = 0
    start_batch = 0
    start_batch_2 = 0
    batch_start += n_steps

    bi_cell_size = 2*cell_size
    bi_cell_size_2 = 2*cell_size_2


    Lamada = 0.001
    drop_keep_rate = 0.15
    lstm_dropout = 0.3


    #variable:
    xs = tf.placeholder(tf.float32, [batch_size, n_steps, input_size])
    # word_embedded_train_tf = tf.reshape(word_embedded_train_tf, [-1, n_steps, input_size])
    ys = tf.placeholder(tf.float32, [None,output_size_2])


# add input layer
    with tf.variable_scope('in_hidden'):
        l_in_x = tf.reshape(xs, [-1, input_size],
                            name='2_2D')  # (batch, n_step, in_size).>>>>(batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = _weight_variable([input_size, cell_size])
        # bs (cell_size, )
        bs_in = _bias_variable([cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        l_in_y = tf.reshape(l_in_y, [-1, n_steps, cell_size], name='2_3D')



# add cell layer1
    with tf.variable_scope('LSTM_cell'):
        outputs= Bidirectional(
                LSTM(cell_size, return_sequences=True, kernel_regularizer=regularizers.l2(0.01),
                 recurrent_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l2(0.01),dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(l_in_y)

        outputs = tf.nn.dropout(outputs, drop_keep_rate)
    #cell_outputs_1[:, -1, :]这个是关键的改进点!
    # with tf.variable_scope('Max_Pooling_2'):
        # doc_embedded = GlobalMaxPool1D()(outputs)
        # doc_embedded = GlobalAveragePooling1D(outputs)
        # doc_embedded = attention(outputs, self.attention_dim, self.l2_reg_lambda)

# add_output_layer(  ):
    # shape = (batch * steps, cell_size)
    # shape = (batch * steps, cell_size)
    with tf.variable_scope('out_hidden'):
        sentence_embedded = outputs[:, -1, :]
        l_out_x = tf.reshape(sentence_embedded, [-1, bi_cell_size], name='2_2D')
        Ws_out = _weight_variable([bi_cell_size, output_size])
        bs_out = _bias_variable([output_size, ])
        # shape = (batch * steps_2, output_size_2)
        with tf.name_scope('Wx_plus_b'):
            l_in_y_2 = tf.matmul(l_out_x, Ws_out) + bs_out
        l_in_y_2 = tf.reshape(l_in_y_2, [-1, n_steps_2, cell_size_2], name='2_3D')

    # xs_2= tf.placeholder(tf.float32, [None, n_steps_2, input_size_2])


# add cell layer2
    with tf.variable_scope('LSTM_cell_2'):
        outputs_2= Bidirectional(
                LSTM(cell_size_2, return_sequences=True, kernel_regularizer=regularizers.l2(0.01),
                 recurrent_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l2(0.01),dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(l_in_y_2)

        outputs_2 =  tf.nn.dropout(outputs_2, drop_keep_rate)
    # add_output_layer(  ):
    # shape = (batch * steps, cell_size)
    # shape = (batch * steps, cell_size)
    with tf.variable_scope('out_hidden_2'):
        doc_embedded = outputs_2[:, -1, :]
        Ws_out_2 = _weight_variable([bi_cell_size_2, output_size_2])
        bs_out_2 = _bias_variable([output_size_2, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(doc_embedded, Ws_out_2) + bs_out_2

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("logs", sess.graph)
    for j in range(50):
        print('epoch j/50: ', j)
        # for i in range(27):
        correct_num_all = 0
        for i in range(110):
            print('i: ', i)
            with tf.name_scope('cost'):
                tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
                regularization_cost = Lamada * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数

                original_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys))

                cost = original_cost_function + regularization_cost

                tf.summary.scalar('cost', cost)

            with tf.name_scope('train'):
                train_op = tf.train.AdamOptimizer(LR).minimize(cost)

            # training
            # word_embedded_train_np = word_embedded_train.eval(session=sess)


            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)



            word_embedded_train_batch = word_embedded_train[i * (start_batch + batch_size) * n_steps:(i + 1) * (
                        start_batch + batch_size) * n_steps, :]
            one_hot_train_labels_batch = one_hot_train_labels[
                                         i * (start_batch_2 + batch_size_2):(i + 1) * (start_batch_2 + batch_size_2), :]

            word_embedded_train_batch = word_embedded_train_batch.reshape((batch_size, n_steps, input_size))

            feed_dict = {xs: word_embedded_train_batch, ys: one_hot_train_labels_batch}

            _, cost= sess.run([train_op,cost], feed_dict)
            if i % 1 == 0:
                print('cost: ', cost)
            with tf.name_scope("accuracy"):
                prediction = tf.argmax(logits, 1)
                real = tf.argmax(one_hot_train_labels_batch, 1)
                prediction = sess.run(prediction, feed_dict)
                correct_prediction = tf.equal(prediction, real)  # ys 需要设定一下
                correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                correct_num_ = sess.run(correct_num, feed_dict)
                correct_num_all += correct_num_
                print('correct_num_all', correct_num_all)
                print('correct_num', correct_num_)
        accuracy = correct_num_all / len(one_hot_train_labels)
        print("cost j/50 :", cost)
        print("Accuracy score:", accuracy)

        correct_num_val_all = 0
        for k in range(11):
            with tf.name_scope('cost'):
                tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
                regularization_cost = Lamada * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数

                original_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys))

                cost = original_cost_function + regularization_cost

                tf.summary.scalar('cost', cost)


            # training
            # word_embedded_train_np = word_embedded_train.eval(session=sess)


            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            word_embedded_val_batch = word_embedded_val[k * (start_batch + batch_size) * n_steps:(k + 1) * (
                        start_batch + batch_size) * n_steps, :]
            one_hot_val_labels_batch = one_hot_val_labels[
                                         k * (start_batch_2 + batch_size_2):(k + 1) * (start_batch_2 + batch_size_2), :]

            word_embedded_val_batch = word_embedded_val_batch.reshape((batch_size, n_steps, input_size))

            feed_dict_val = {xs: word_embedded_val_batch, ys: one_hot_val_labels_batch}

            cost_val = sess.run( cost, feed_dict_val)
            print('set done')
            if k % 1 == 0:
                print('cost: ', cost_val)
            with tf.name_scope("accuracy_val"):
                prediction_val = tf.argmax(logits, 1)
                real_val = tf.argmax(one_hot_val_labels_batch, 1)
                prediction_val = sess.run(prediction_val, feed_dict_val)
                correct_prediction_val = tf.equal(prediction_val, real_val)  # ys 需要设定一下
                correct_num_val = tf.reduce_sum(tf.cast(correct_prediction_val, tf.float32))
                correct_num_val = sess.run(correct_num_val, feed_dict_val)
                correct_num_val_all += correct_num_val
                print('correct_num_all', correct_num_val_all)
                print('correct_num', correct_num_val)
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            # accuracy = sess.run(accuracy, feed_dict)
        accuracy_val = correct_num_val_all / len(one_hot_val_labels)
        print("cost_val j/50:", cost_val)
        print("Accuracy score:", accuracy_val)
        # plt.plot(j, cost,'r')
        # plt.plot(j, cost_val, 'g')
        # plt.plot(j, accuracy, 'p')
        # plt.plot(j, accuracy_val, 'b')
        # plt.show()





            # if i == 0:
            #     feed_dict = {xs: word_embedded_train_batch, ys: one_hot_train_labels_batch}
            #
            # else:
            #     feed_dict = {xs: word_embedded_train_batch, ys: one_hot_train_labels_batch,
            #                  cell_init_state_fw: final_state_f ,cell_init_state_bw: final_state_b,
            #                  cell_init_state_fw_2: final_state_f_2 ,cell_init_state_bw_2: final_state_b_2}


            # _, cost, final_state_f,final_state_b, final_state_f_2,final_state_b_2, pred = sess.run(
            #             #     [train_op, cost, final_state_fw,final_state_bw,final_state_fw_2,final_state_bw_2, prediction],
            #             #     feed_dict=feed_dict)

        #     feed_dict = {xs: word_embedded_train_batch, ys: one_hot_train_labels_batch,
        #                  cell_init_state_fw: final_state_f, cell_init_state_bw: final_state_b,
        #                  cell_init_state_fw_2: final_state_f_2, cell_init_state_bw_2: final_state_b_2}
        # _, cost, outputs_state_f_c, outputs_state_b_c, outputs_state_f_h, outputs_state_b_h, outputs_state_f_2_c, outputs_state_b_2_c, outputs_state_f_2_h, outputs_state_b_2_h, pred = sess.run(
        #     [train_op, cost, outputs_state_fw.c, outputs_state_bw.c, outputs_state_fw.h, outputs_state_bw.h,
        #      outputs_state_fw_2.c, outputs_state_bw_2.c, outputs_state_fw_2.h, outputs_state_bw_2.h, prediction],
        #     feed_dict=feed_dict)
        #
        # final_state_f = tf.concat((outputs_state_f_c, outputs_state_f_h), 1)
        # final_state_b = tf.concat((outputs_state_b_c, outputs_state_b_h), 1)
        # final_state_f_2 = tf.concat((outputs_state_f_2_c, outputs_state_f_2_h), 1)
        # final_state_b_2 = tf.concat((outputs_state_b_2_c, outputs_state_b_2_h), 1)



            # word_embedded_val_batch = word_embedded_val[i * (start_batch + batch_size) * n_steps:(i + 1) * (
            #             start_batch + batch_size) * n_steps, :]
            # one_hot_val_labels_batch = one_hot_val_labels[
            #                              i * (start_batch_2 + batch_size_2):(i + 1) * (start_batch_2 + batch_size_2), :]
            #
            # word_embedded_val_batch = word_embedded_val_batch.reshape((batch_size, n_steps, input_size))

            # if 'final_s_' not in globals():  # first state, no any hidden state
            #     feed_dict = {xs: word_embedded_train_batch, ys: one_hot_train_labels_batch}
            # else:  # has hidden state, so pass it to rnn

            # feed_dict_val = {xs: word_embedded_val_batch, ys: one_hot_val_labels_batch,
            #              cell_init_state_2: cell_final_state_2, cell_final_state: cell_init_state}

            # _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)

                # to visualize the result and improvement
                # try:
                #     ax.lines.remove(lines[0])
                # except Exception:
                #     passmain_2_test.py:520main_2_test.py:520
                # prediction_value = sess.run(prediction, feed_dict={xs: word_embedded_train})
                # # plot the prediction
                # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

            # start_batch += batch_size
            # start_batch_2 += batch_size_2

        # with tf.name_scope("accuracy"):
        #     prediction = tf.argmax(logits, 1)
        #     real= tf.argmax(one_hot_train_labels_batch, 1) #return the raw maxvalue's index
        #     prediction, real = sess.run([prediction, real], feed_dict)
        #     # prediction = sess.run(prediction, feed_dict)
        #     # # real= sess.run(real, feed_dict)
        #     correct_prediction = tf.equal(prediction, real)  # ys 需要设定一下
        #     correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #     correct_num_ = sess.run(correct_num, feed_dict)
        #     correct_num_all += correct_num_
        #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        #     # accuracy = sess.run(accuracy, feed_dict)
        #     accuracy = correct_num_all/len(one_hot_train_labels)
        #     # accuracy_val = sess.run(accuracy, feed_dict_val)
        #     print("Accuracy score:", accuracy)



    # if  keep_prob < 1:
    #     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    #         lstm_cell, output_keep_prob= keep_prob
    #     )
    #

    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * hidden_layer_num, state_is_tuple=True)
    #
    #
    # bi_lstm_word_sent = Bidirectional(GRU(lstm_unit_size, return_sequences=False))(word_embedded)
    #
    # # attention_word = AttentionWithContext()(bi_lstm_word_sent)
    #
    # word_sent_encode = Dropout(dropout)(bi_lstm_word_sent)  # (attention_word)
    # word_encoder = Model(inputs=sent_word_input, outputs=word_sent_encode)
    # word_encoded = TimeDistributed(word_encoder)(doc_word_input)
    #
    # b_lstm_word_doc = Bidirectional(GRU(lstm_unit_size, return_sequences=False))(word_encoded)
    #
    # word_output = Dropout(dropout, name='final_layer_word')(b_lstm_word_doc)  # (attention_doc)
    # word_output = Dense(unique_labels_count, activation='softmax')(word_output)  # (word_output)
    #
    # model = Model(inputs=doc_word_input, outputs=word_output)
    # rmsprop = RMSprop(lr=learning_rate)
    # model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #
    # final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_word').output)
    #
    # model.summary()
    # model.fit(x_train.index, one_hot_train_labels,
    #           validation_data=(x_val.index, one_hot_val_labels), epochs=epoch_num,
    #           verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
    # evaluation = final_layer.evaluate(val_word_input, one_hot_val_labels, verbose=0)
    # nn_predictions = final_layer.predict(word_sequence_matrix, verbose=0)
    #
    # feature_matrix_2 = DataFrame(nn_predictions)

    # print("Training Classifier")
    # start = time()
    # clf.fit(feature_matrix_2.loc[x_train.index.append(x_val.index)], labels_series[x_train.index.append(x_val.index)])
    # print("Trained classifier in", time() - start, "seconds")
    #
    # print("Predicting test set")
    # predictions = clf.predict(feature_matrix_2.loc[x_test.index])
    # print(classification_report(y_test, predictions))
    # print("Accuracy score:", accuracy_score(y_test, predictions))
    # print("Micro F1:", f1_score(y_test, predictions, average='micro'))
    # print("Macro F1:", f1_score(y_test, predictions, average='macro'))
    # print("Neural Network Loss:", evaluation[0])
    # print("Neural Network Accuracy:", evaluation[1])


if __name__ == '__main__':
    print(args)
    print("Detected", cpu_count(), "CPU cores")
    main()
