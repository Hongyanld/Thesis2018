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
from itertools import product
from multiprocessing import cpu_count
from random import randint, shuffle

import numpy as np
from gensim.models import Word2Vec
from keras import Input, Model
from keras import callbacks
from keras.backend import clear_session
from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTM
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from pandas import read_table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = ArgumentParser()
parser.add_argument('type', metavar='TYPE', type=str, help='TYPE can be one of the following: "pos", "word", "char"')
parser.add_argument('unit', metavar='INT', type=int, help='Unit size for LSTM')
args = parser.parse_args()

root_path = "/home/werfelrk/"
word_embeddings_size = 300
max_num_of_sentences = 25
max_num_of_tokens_per_sentence = 50  # Covers about 98% of data
max_num_of_chars = 300


def read_data(file='toefl11_tokenized.tsv'):
    print("Reading index from", file)
    return read_table(file)


def train_val_test_split(df):

    x_train, x_test, y_train, y_test = train_test_split(df, df['Language'], stratify=df[['Language', 'Prompt']],
                                                        test_size=2200, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, x_test['Language'], stratify=x_test[['Language', 'Prompt']],
                                                    test_size=1100, random_state=0)
    return x_train, x_val, y_train, y_val


def main():
    df = read_data()

    labels_series = df['Language']
    unique_labels_count = len(Counter(labels_series))

    if args.type == "word":
        w2v_model = Word2Vec.load(root_path + "model/toefl11.word_model.bin")
        with open(root_path + "model/toefl11_tokenized.tsv.word_sequence_matrix.pickle", "rb") as file_to_read:
            sequence_matrix = pickle.load(file_to_read)
    elif args.type == "char":
        w2v_model = Word2Vec.load(root_path + "model/toefl11.char_model.bin")
        with open(root_path + "model/toefl11_tokenized.tsv.char_sequence_matrix.pickle", "rb") as file_to_read:
            sequence_matrix = pickle.load(file_to_read)
    else:
        w2v_model = Word2Vec.load(root_path + "model/toefl11.pos_model.bin")
        with open(root_path + "model/toefl11_tokenized.tsv.pos_sequence_matrix.pickle", "rb") as file_to_read:
            sequence_matrix = pickle.load(file_to_read)

    token_indices = dict((p, i) for i, p in enumerate(w2v_model.wv.vocab, start=1))

    encoder = LabelEncoder().fit(labels_series)
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto')
    # history = LossHistory()

    x_train, x_test, y_train, y_test = train_val_test_split(df)

    one_hot_train_labels = to_categorical(encoder.transform(y_train), num_classes=unique_labels_count)
    one_hot_test_labels = to_categorical(encoder.transform(y_test), num_classes=unique_labels_count)

    train_input = sequence_matrix[x_train.index]
    test_input = sequence_matrix[x_test.index]

    epoch_num = 50

    lstm_unit_sizes = [args.unit]  # set([64 * randint(1, 16) for _ in range(1, 6, 1)])
    learning_rates = [2e-4+2e-4*i for i in range(6)]  # set(np.random.uniform(2e-4, 2e-3, 10))
    dropouts = [5*i/100 for i in range(13, 16)]  # set(np.random.uniform(1e-1, 1e0-0.1, 5))
    params = list(product(lstm_unit_sizes, learning_rates, dropouts))
    shuffle(params)
    best_accuracy = 0
    best_conf = []
    for size, lr, drop in params:
        word_symbols = len(token_indices) + 1
        word_embedding_weights = np.zeros((word_symbols, word_embeddings_size))
        for word, index in token_indices.items():
            word_embedding_weights[index, :] = w2v_model[word]

        if args.type == "char":
            doc_word_input = Input(shape=(max_num_of_sentences, max_num_of_chars), dtype='int64')
            sent_word_input = Input(shape=(max_num_of_chars,), dtype='int64')
        else:
            doc_word_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64')
            sent_word_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64')

        word_embedding_layer = Embedding(output_dim=word_embeddings_size, input_dim=word_symbols, mask_zero=True)
        word_embedding_layer.build((None,))  # if you don't do this, the next step won't work
        word_embedding_layer.set_weights([word_embedding_weights])
        word_embedded = word_embedding_layer(sent_word_input)

        bi_lstm_word_sent = Bidirectional(LSTM(size, return_sequences=False))(word_embedded)

        word_sent_encode = Dropout(drop)(bi_lstm_word_sent)
        word_encoder = Model(inputs=sent_word_input, outputs=word_sent_encode)
        word_encoded = TimeDistributed(word_encoder)(doc_word_input)

        b_lstm_word_doc = Bidirectional(LSTM(size, return_sequences=False))(word_encoded)

        word_output = Dropout(drop)(b_lstm_word_doc)
        word_output = Dense(unique_labels_count, activation='softmax')(word_output)

        model = Model(inputs=doc_word_input, outputs=word_output)
        rmsprop = RMSprop(lr=lr)
        model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_word').output)
        model.fit(train_input, one_hot_train_labels,
                  validation_data=(test_input, one_hot_test_labels), epochs=epoch_num,
                  verbose=0, batch_size=32, shuffle=True, callbacks=[earlystop_cb, check_cb])
        evaluation = model.evaluate(test_input, one_hot_test_labels, verbose=0)
        print("lstm_unit_size=%d, learning_rate=%f, dropout=%f" % (size, lr, drop), "prediction:", evaluation)
        if evaluation[1] > best_accuracy:
            print("new best accuracy")
            best_accuracy = evaluation[1]
            best_conf = [size, lr, drop]
        clear_session()
    print("Best accuracy:", best_accuracy)
    print("Best configuration:", best_conf)


if __name__ == '__main__':
    if args.type != 'word' and args.type != 'char' and args.type != 'pos':
        parser.print_help()
        exit(1)
    print(args)
    print("Detected", cpu_count(), "CPU cores")
    main()
