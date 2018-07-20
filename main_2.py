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
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTM, Lambda, Concatenate, \
    MaxPooling1D, Conv1D, GlobalMaxPool1D, concatenate, GRU
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


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
        print("Loading word2vec models from", root_path + "model_2/")
        word_model = Word2Vec.load(root_path + "model_2/toefl11.word_model.bin")
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

    print("\tSaving word2vec models to", root_path + "model_2/")
    word_model.save(root_path + "model_2/toefl11.word_model.bin")

    return word_model

def train_word_vector(word_model):
    vocab = word_model.wv.vocab
    word_vector = {}
    for word in vocab:
        word_vector[word] = word_model[word]
    word_vector_add = {0: np.zeros(300)};
    word_vector.update(word_vector_add)
    print("\tSaving word_vector to", root_path + "model_2/")
    with open(root_path + "model_2/"  + "text_series.word_vector.pickle", "wb") as file_to_write:
        pickle.dump(word_vector, file_to_write, protocol=4)
    return word_vector

def get_word_vector(word_model):
    if args.word_vector: #?
        print("Loading word_vector from", root_path + "model_2/")
        with open(root_path + "model_2/"  + "text_series.word_vector.pickle", "rb") as file_to_read:
            word_vector = pickle.load(file_to_read)
    else:
        word_vector = train_word_vector(word_model)
    return word_vector


def get_feature_matrix(text_series, word_indices):
    if args.feature_matrix:
        print("Loading feature matrices from", root_path + "model_2/")
        with open(root_path + "model_2/" + args.data_file + ".word_sequence_matrix.pickle", "rb") as file_to_read:
            word_sequence_matrix = pickle.load(file_to_read)
    else:
        print("\tBuilding Word sequence matrix")
        word_sequence_matrix = np.stack([build_sentence_word_sequences(
            text_series[index], word_indices) for index in range(len(text_series))])
        with open(root_path + "model_2/" + args.data_file + ".word_sequence_matrix.pickle", "wb") as file_to_write:
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
#         print("Loading text_word_sequence_embedded from", root_path + "model_2/")
#         with open(root_path + "model_2/" + args.data_file + ".doc_sequence_embedded.pickle", "rb") as file_to_read:
#             doc_sequence_embedded = pickle.load(file_to_read)
#     else:
#         print("\tBuilding doc_sequence_embedded")
#         word_sequence_matrix = get_feature_matrix(text_series,word_indices)
#         doc_sequence_embedded = np.stack(
#             (word_vector[word_indices_inv[word_sequence_matrix[test_series[index], i, j]]]) for index in range(0, 12100)
#             for i in range(max_num_of_sentences) for j in range(max_num_of_tokens_per_sentence))
#         with open(root_path + "model_2/" + args.data_file + ".doc_sequence_embedded.pickle", "wb") as file_to_write:
#             pickle.dump(text_word_sequence_embedded, file_to_write, protocol=4)
#     return doc_sequence_embedded

def get_spelling_errors(text):
    return " ".join([err.word for err in SpellChecker("en_US", text)]) or ""


# record history of training
class LossHistory(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.logs = {}
        self.accuracies = []
        self.losses = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))



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
                                                    test_size=0.1, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test

def word_sequence_GRU_model(word_indices, word_model, unique_labels_count):
    data_dim = word_embeddings_size
    timesteps = 50
    num_classes = unique_labels_count
    batch_size = 25




    #word_embedding_weights is the (index, vector) of words vocabulary
    word_symbols = len(word_indices) + 1
    word_embedding_weights = np.zeros((word_symbols, word_embeddings_size))

    for word, index in word_indices.items():
        try:
            word_embedding_weights[index, :] = word_model[word]
        except KeyError:
            word_embedding_weights[index, :] = np.ones(word_embeddings_size) * -1


    # word_model["some"] return the vector of data "some"
    # word_data[0] is the data for text0 with the index of 0.

    sent_word_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64', name="sent_word_input")

    word_embedding_layer = Embedding(output_dim=word_embeddings_size, input_dim=word_symbols, mask_zero=True)
    word_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    word_embedding_layer.set_weights([word_embedding_weights])
    word_embedded = word_embedding_layer(sent_word_input)

    model = Sequential()(word_embedded)
    model.add(LSTM(lstm_unit_size, return_sequences=True,
                   input_shape=(timesteps, word_embeddings_size)))
    model.add(LSTM(lstm_unit_size, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, timesteps, lstm_unit_size)))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def main():
    df = read_data(args.data_file)
    #df2 = read_data(args.data_file_a)
    # df = df[df.Language != 'Hindi']
    # df = df[df.Language != 'Telugu']
    # df.drop(columns='Prompt', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    labels_series = df['Language']
    text_series = df['Text']
    #text_matrix = df2['Text']
    unique_labels_count = len(Counter(labels_series))

    word_model= get_word2vec_model(text_series)

    word_vector = get_word_vector(word_model)

    word_indices = dict((p, i) for i, p in enumerate(word_model.wv.vocab, start=1))
    word_indices_inv = dict((p, i) for p, i in enumerate(word_model.wv.vocab, start=1))
    # word_indices_inv_add = {0: 0};
    # word_indices_inv.update(word_indices_inv_add)

    word_sequence_matrix = get_feature_matrix(text_series,word_indices)
    # doc_series_embedded = np.stack(
    #     [word_vector.get(word_indices_inv.setdefault(word_sequence_matrix[index, i, j], 0)) for index in range(0, 12100)
    #      for i in range(max_num_of_sentences) for j in range(max_num_of_tokens_per_sentence)])

    if args.doc_sequence_embedded:
        print("Loading text_word_sequence_embedded from", root_path + "model_2/")
        with open(root_path + "model_2/" + args.data_file + ".doc_sequence_embedded.pickle", "rb") as file_to_read:
            doc_sequence_embedded = pickle.load(file_to_read)
    else:
        print("\tBuilding doc_sequence_embedded")
        doc_sequence_embedded = np.stack(
            [word_vector.get(word_indices_inv.setdefault(word_sequence_matrix[index, i, j],0)) for index in range(0, len(text_series))
            for i in range(max_num_of_sentences) for j in range(max_num_of_tokens_per_sentence)])
        with open(root_path + "model_2/" + args.data_file + ".doc_sequence_embedded.pickle", "wb") as file_to_write:
            pickle.dump(doc_sequence_embedded, file_to_write, protocol=4)


    encoder = LabelEncoder().fit(labels_series)
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    # history = LossHistory()

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(df)


    # define input and output
    one_hot_train_labels = to_categorical(encoder.transform(y_train), num_classes=unique_labels_count)
    one_hot_val_labels = to_categorical(encoder.transform(y_val), num_classes=unique_labels_count)

    train_word_input = doc_sequence_embedded([x_train.index])

    val_word_input = doc_sequence_embedded[x_val.index]

# start model train:

    epoch_num = 50
    print("Training neural network for", epoch_num, "epoch(s)")
    word_sequence_GRU_model1(word_indices, word_model, unique_labels_count)
    model, final_layer = word_sequence_GRU_model(word_indices, word_model, unique_labels_count)
    model.summary()
    model.fit(train_word_input, one_hot_train_labels,
                 validation_data=(val_word_input, one_hot_val_labels), epochs=epoch_num,
                 verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
    evaluation = final_layer.evaluate(val_word_input, one_hot_val_labels, verbose=0)
    nn_predictions = final_layer.predict(word_sequence_matrix, verbose=0)

    feature_matrix_2 =DataFrame(nn_predictions)

    print("Training Classifier")
    start = time()
    clf.fit(feature_matrix_2.loc[x_train.index.append(x_val.index)], labels_series[x_train.index.append(x_val.index)])
    print("Trained classifier in", time() - start, "seconds")

    print("Predicting test set")
    predictions = clf.predict(feature_matrix_2.loc[x_test.index])
    print(classification_report(y_test, predictions))
    print("Accuracy score:", accuracy_score(y_test, predictions))
    print("Micro F1:", f1_score(y_test, predictions, average='micro'))
    print("Macro F1:", f1_score(y_test, predictions, average='macro'))
    print("Neural Network Loss:", evaluation[0])
    print("Neural Network Accuracy:", evaluation[1])


if __name__ == '__main__':
    print(args)
    print("Detected", cpu_count(), "CPU cores")
    # main()
