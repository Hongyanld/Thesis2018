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

import keras_resnet
import keras_resnet.blocks
import keras_resnet.models
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTM, Lambda, Concatenate, \
    MaxPooling1D, Conv1D, GlobalMaxPool1D, concatenate, GRU
from keras.optimizers import RMSprop

from main import word_embeddings_size, pos_embeddings_size, char_embeddings_size, args, max_num_of_sentences, \
    max_num_of_tokens_per_sentence, max_num_of_chars
from AttentionWithContext import AttentionWithContext


def binarize_pos(x, sz=44):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_pos_outshape(in_shape):
    return in_shape[0], in_shape[1], 44


def binarize_char(x, sz=93):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_char_outshape(in_shape):
    return in_shape[0], in_shape[1], 93


def build_resnet_model(word_indices, word_vectors, output_dim):  # Currently does not work
    learning_rate = args.learning_rate
    dropout = 0.5

    word_symbols = len(word_indices) + 1
    word_embedding_weights = np.zeros((word_symbols, word_embeddings_size))
    for word, index in word_indices.items():
        word_embedding_weights[index, :] = word_vectors[word]

    doc_word_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64',
                           name="doc_word_input")
    sent_word_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64',
                            name="sent_word_input")

    word_embedding_layer = Embedding(output_dim=word_embeddings_size, input_dim=word_symbols, mask_zero=True)
    word_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    word_embedding_layer.set_weights([word_embedding_weights])
    word_embedded = word_embedding_layer(sent_word_input)

    blocks = [2, 2, 2, 2]
    block = keras_resnet.blocks.basic_1d

    resnet_word_sent = keras_resnet.models.ResNet50(word_embedded, blocks, block, classes=output_dim)

    word_sent_encode = Dropout(dropout)(resnet_word_sent)
    word_encoder = Model(inputs=sent_word_input, outputs=word_sent_encode)
    word_encoded = TimeDistributed(word_encoder)(doc_word_input)

    resnet_word_doc = keras_resnet.models.ResNet(word_encoded, blocks, block, classes=output_dim)

    word_output = Dropout(dropout)(resnet_word_doc)
    word_output = Dense(output_dim, activation='softmax')(word_output)

    model = Model(inputs=doc_word_input, outputs=word_output)
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def build_pos_sequence_lstm_model(pos_indices, pos_vectors, output_dim):
    lstm_unit_size = args.lstm_unit_size
    learning_rate = args.learning_rate
    lstm_dropout = 0.15
    dropout = 0.3

    doc_pos_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64',
                          name="doc_pos_input")
    sent_pos_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64', name="sent_pos_input")

    pos_symbols = len(pos_indices) + 1
    pos_embedding_weights = np.zeros((pos_symbols, pos_embeddings_size))
    for pos, index in pos_indices.items():
        pos_embedding_weights[index, :] = pos_vectors[pos]

    pos_embedding_layer = Embedding(output_dim=pos_embeddings_size, input_dim=pos_symbols, mask_zero=True)
    pos_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    pos_embedding_layer.set_weights([pos_embedding_weights])
    pos_embedded = pos_embedding_layer(sent_pos_input)

    # filter_length = [5, 3, 3]
    # nb_filter = [196, 196, 256]
    # pool_length = 2
    # pos_embedded = Lambda(binarize_pos, output_shape=binarize_pos_outshape)(sent_pos_input)
    # for i in range(len(nb_filter)):
    #     pos_embedded = Conv1D(filters=nb_filter[i], kernel_size=filter_length[i], padding='valid', activation='relu',
    #                           kernel_initializer='glorot_normal', strides=1)(pos_embedded)
    #     pos_embedded = Dropout(0.1)(pos_embedded)
    #     pos_embedded = MaxPooling1D(pool_size=pool_length)(pos_embedded)

    bi_lstm_pos_sent = Bidirectional(
        LSTM(lstm_unit_size, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(
        pos_embedded)

    pos_sent_encode = Dropout(dropout)(bi_lstm_pos_sent)
    pos_encoder = Model(inputs=sent_pos_input, outputs=pos_sent_encode)

    pos_encoded = TimeDistributed(pos_encoder)(doc_pos_input)

    b_lstm_pos_doc = Bidirectional(
        LSTM(lstm_unit_size, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(pos_encoded)

    pos_output = Dropout(dropout, name='final_layer_pos')(b_lstm_pos_doc)
    pos_output = Dense(output_dim, activation='softmax')(pos_output)

    model = Model(inputs=doc_pos_input, outputs=pos_output)
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_pos').output)

    return model, final_layer


def build_word_sequence_lstm_model(word_indices, word_vectors, output_dim):
    lstm_unit_size = args.lstm_unit_size
    learning_rate = args.learning_rate
    dropout = 0.6
 # from word_indices of all word vocabulary to word_embedded for one sentence of text
    word_symbols = len(word_indices) + 1
    word_embedding_weights = np.zeros((word_symbols, word_embeddings_size))
    for word, index in word_indices.items():
        try:
            word_embedding_weights[index, :] = word_vectors[word]
        except KeyError:
            word_embedding_weights[index, :] = np.ones(word_embeddings_size) * -1

    doc_word_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64',
                           name="doc_word_input")
    sent_word_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64', name="sent_word_input")

    word_embedding_layer = Embedding(output_dim=word_embeddings_size, input_dim=word_symbols, mask_zero=True)
    word_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    word_embedding_layer.set_weights([word_embedding_weights])
    word_embedded = word_embedding_layer(sent_word_input)



    bi_lstm_word_sent = Bidirectional(GRU(lstm_unit_size, return_sequences=rueT))(word_embedded)


    #attention_word = AttentionWithContext()(bi_lstm_word_sent)

    word_sent_encode = Dropout(dropout)(bi_lstm_word_sent)#(attention_word)
    word_encoder = Model(inputs=sent_word_input, outputs=word_sent_encode)
    word_encoded = TimeDistributed(word_encoder)(doc_word_input)

    b_lstm_word_doc = Bidirectional(GRU(lstm_unit_size, return_sequences=False))(word_encoded)

    word_output = Dropout(dropout, name='final_layer_word')(b_lstm_word_doc)#(attention_doc)
    word_output = Dense(output_dim, activation='softmax')(word_output)#(word_output)

    model = Model(inputs=doc_word_input, outputs=word_output)
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_word').output)
    # final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_word').output)

    return model, final_layer


def build_char_sequence_lstm_model(char_indices, char_vectors, output_dim):
    lstm_unit_size = args.lstm_unit_size
    learning_rate = args.learning_rate
    lstm_dropout = 0.15
    dropout = 0.3

    doc_char_input = Input(shape=(max_num_of_sentences, max_num_of_chars), dtype='int64', name="doc_char_input")
    sent_char_input = Input(shape=(max_num_of_chars,), dtype='int64', name="sent_char_input")

    char_symbols = len(char_indices) + 1
    char_embedding_weights = np.zeros((char_symbols, char_embeddings_size))
    for char, index in char_indices.items():
        char_embedding_weights[index, :] = char_vectors[char]
    char_embedding_layer = Embedding(output_dim=char_embeddings_size, input_dim=char_symbols, mask_zero=True)
    char_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    char_embedding_layer.set_weights([char_embedding_weights])
    char_embedded = char_embedding_layer(sent_char_input)

    # filter_length = [5, 3, 3]
    # nb_filter = [196, 196, 256]
    # pool_length = 2
    # char_embedded = Lambda(binarize_char, output_shape=binarize_char_outshape)(sent_char_input)
    # for i in range(len(nb_filter)):
    #     char_embedded = Conv1D(filters=nb_filter[i], kernel_size=filter_length[i], padding='valid', activation='relu',
    #                            kernel_initializer='glorot_normal', strides=1)(char_embedded)
    #
    #     char_embedded = Dropout(0.1)(char_embedded)
    #     char_embedded = MaxPooling1D(pool_size=pool_length)(char_embedded)

    bi_lstm_char_sent = Bidirectional(
        LSTM(lstm_unit_size, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(
        char_embedded)

    char_sent_encode = Dropout(dropout)(bi_lstm_char_sent)
    char_encoder = Model(inputs=sent_char_input, outputs=char_sent_encode)
    char_encoded = TimeDistributed(char_encoder)(doc_char_input)

    b_lstm_char_doc = Bidirectional(
        LSTM(lstm_unit_size, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(
        char_encoded)

    char_output = Dropout(dropout, name='final_layer_char')(b_lstm_char_doc)
    char_output = Dense(output_dim, activation='softmax')(char_output)

    model = Model(inputs=doc_char_input, outputs=char_output)
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_char').output)

    return model, final_layer


def build_combined_lstm_model(word_indices, pos_indices, char_indices, w2v_word_model, w2v_pos_model, w2v_char_model,
                              output_dim):
    learning_rate = args.learning_rate

    pos_model, _ = build_pos_sequence_lstm_model(pos_indices, w2v_pos_model, output_dim=64)
    word_model, _ = build_word_sequence_lstm_model(word_indices, w2v_word_model, output_dim=64)
    char_model, _ = build_char_sequence_lstm_model(char_indices, w2v_char_model, output_dim=64)

    mrg = Concatenate(name='final_layer_mrg')([pos_model.output, word_model.output, char_model.output])
    dense = Dense(output_dim, activation='softmax')(mrg)

    model = Model(inputs=[pos_model.input, word_model.input, char_model.input], outputs=dense)
    print("\tLearning rate:", learning_rate)
    rmsprop = RMSprop(lr=learning_rate)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_mrg').output)

    return model, final_layer


def char_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


def build_pos_sequence_conv_model(output_dim):
    pos_doc_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64')
    pos_sent_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64')

    pos_embedded = Lambda(binarize_pos, output_shape=binarize_pos_outshape)(pos_sent_input)

    block2 = char_block(pos_embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = char_block(pos_embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))

    sent_encode = concatenate([block2, block3], axis=-1)
    # sent_encode = Dropout(0.2)(sent_encode)

    encoder = Model(inputs=pos_sent_input, outputs=sent_encode)
    encoder.summary()

    encoded = TimeDistributed(encoder)(pos_doc_input)

    lstm_h = 92

    lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0,
                       name='final_layer_pos')(lstm_layer)

    # output = Dropout(0.2)(bi_lstm)
    output = Dense(output_dim, activation='softmax')(lstm_layer2)

    model = Model(outputs=output, inputs=pos_doc_input)
    optimizer = 'rmsprop'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_pos').output)

    return model, final_layer


def build_word_sequence_conv_model(word_indices, word_vectors, output_dim):
    word_symbols = len(word_indices) + 1
    word_embedding_weights = np.zeros((word_symbols, word_embeddings_size))
    for word, index in word_indices.items():
        word_embedding_weights[index, :] = word_vectors[word]

    word_doc_input = Input(shape=(max_num_of_sentences, max_num_of_tokens_per_sentence), dtype='int64')
    word_sent_input = Input(shape=(max_num_of_tokens_per_sentence,), dtype='int64')

    word_embedding_layer = Embedding(output_dim=word_embeddings_size, input_dim=word_symbols, mask_zero=False)
    word_embedding_layer.build((None,))  # if you don't do this, the next step won't work
    word_embedding_layer.set_weights([word_embedding_weights])
    word_embedded = word_embedding_layer(word_sent_input)

    block2 = char_block(word_embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = char_block(word_embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))

    sent_encode = concatenate([block2, block3], axis=-1)
    # sent_encode = Dropout(0.2)(sent_encode)

    encoder = Model(inputs=word_sent_input, outputs=sent_encode)
    encoder.summary()

    encoded = TimeDistributed(encoder)(word_doc_input)

    lstm_h = 92

    lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0,
                       name='final_layer_word')(lstm_layer)

    # output = Dropout(0.2)(bi_lstm)
    output = Dense(output_dim, activation='softmax')(lstm_layer2)

    model = Model(outputs=output, inputs=word_doc_input)
    optimizer = 'rmsprop'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_word').output)

    return model, final_layer


def build_char_sequence_conv_model(output_dim):
    char_doc_input = Input(shape=(max_num_of_sentences, max_num_of_chars), dtype='int64')
    char_sent_input = Input(shape=(max_num_of_chars,), dtype='int64')

    embedded = Lambda(binarize_char, output_shape=binarize_char_outshape)(char_sent_input)

    block2 = char_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = char_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))

    sent_encode = concatenate([block2, block3], axis=-1)
    # sent_encode = Dropout(0.2)(sent_encode)

    encoder = Model(inputs=char_sent_input, outputs=sent_encode)
    encoder.summary()

    encoded = TimeDistributed(encoder)(char_doc_input)

    lstm_h = 92

    lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0,
                       name='final_layer_char')(lstm_layer)

    # output = Dropout(0.2)(bi_lstm)
    output = Dense(output_dim, activation='softmax')(lstm_layer2)

    model = Model(outputs=output, inputs=char_doc_input)
    optimizer = 'rmsprop'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_char').output)

    return model, final_layer


def build_combined_conv_model(word_indices, word_vectors, output_dim):
    learning_rate = args.learning_rate

    pos_model, _ = build_pos_sequence_conv_model(output_dim=64)
    word_model, _ = build_word_sequence_conv_model(word_indices, word_vectors, output_dim=64)
    char_model, _ = build_char_sequence_conv_model(output_dim=64)

    mrg = Concatenate(name='final_layer_mrg')([pos_model.output, word_model.output, char_model.output])
    dense = Dense(output_dim, activation='softmax')(mrg)

    model = Model(inputs=[pos_model.input, word_model.input, char_model.input], outputs=dense)
    print("\tLearning rate:", learning_rate)
    rmsprop = RMSprop(lr=learning_rate)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    final_layer = Model(inputs=model.input, outputs=model.get_layer('final_layer_mrg').output)

    return model, final_layer
