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
import matplotlib.pyplot as plt

from model import *

from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument('-m', '--model', action='store_true')
parser.add_argument('-f', '--feature_matrix', action='store_true')
parser.add_argument('-k', '--k_value', metavar='INT', type=int, default=5)
parser.add_argument('-d', '--data_file', metavar='PATH', type=str, default='toefl11_tokenized.tsv')
parser.add_argument('-D', '--data_file_test', metavar='PATH', type=str, default='toefl11_tokenized_test.tsv')
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
        print("Loading word2vec models from", root_path + "model/")
        word_model = Word2Vec.load(root_path + "model/toefl11.word_model.bin")
        pos_model = Word2Vec.load(root_path + "model/toefl11.pos_model.bin")
        char_model = Word2Vec.load(root_path + "model/toefl11.char_model.bin")
    else:
        print("Training word2vec model")
        word_model, pos_model, char_model = train_word2vec_model(text_series)

    return word_model, pos_model, char_model


def train_word2vec_model(text_series):
    print("\tBuilding lists of tokens")
    word_data = [sent.lower().split() for doc in text_series for sent in doc.split('\n') if sent]
    pos_data = [convert_text_to_pos_tags(sent).split() for doc in text_series for sent in doc.split('\n') if sent]
    char_data = [list(sent) for doc in text_series for sent in doc.split('\n') if sent]

    epochs = 70
    print("\tTraining word2vec word model with", epochs, "epochs")
    start = time()
    word_model = Word2Vec(word_data, size=word_embeddings_size, alpha=0.02, window=1, min_count=1,
                          workers=cpu_count(), sg=1, iter=epochs)
    print("\tTrained word2vec word model in", (time() - start) / 60, "minutes")

    epochs = 80
    print("\tTraining word2vec pos model with", epochs, "epochs")
    start = time()
    pos_model = Word2Vec(pos_data, size=pos_embeddings_size, alpha=0.04, window=2, min_count=1,
                         workers=cpu_count(), sg=0, iter=epochs)
    print("\tTrained word2vec pos model in", (time() - start) / 60, "minutes")

    epochs = 90
    print("\tTraining word2vec char model with", epochs, "epochs")
    start = time()
    char_model = Word2Vec(char_data, size=char_embeddings_size, alpha=0.04, window=2, min_count=1,
                          workers=cpu_count(), sg=0, iter=epochs)
    print("\tTrained word2vec char model in", (time() - start) / 60, "minutes")

    print("\tSaving word2vec models to", root_path + "model/")
    word_model.save(root_path + "model/toefl11.word_model.bin")
    pos_model.save(root_path + "model/toefl11.pos_model.bin")
    char_model.save(root_path + "model/toefl11.char_model.bin")

    return word_model, pos_model, char_model


def get_feature_matrix(text_series, word_indices, pos_indices, char_indices):
    if args.feature_matrix:
        print("Loading feature matrices from", root_path + "model/")
        with open(root_path + "model/" + args.data_file + ".additional_features.pickle", "rb") as file_to_read:
            feature_matrix = pickle.load(file_to_read)
        with open(root_path + "model/" + args.data_file + ".pos_sequence_matrix.pickle", "rb") as file_to_read:
            pos_sequence_matrix = pickle.load(file_to_read)
        with open(root_path + "model/" + args.data_file + ".word_sequence_matrix.pickle", "rb") as file_to_read:
            word_sequence_matrix = pickle.load(file_to_read)
        with open(root_path + "model/" + args.data_file + ".char_sequence_matrix.pickle", "rb") as file_to_read:
            char_sequence_matrix = pickle.load(file_to_read)
    else:
        # min_f = 0.0005
        # max_f = 0.20
        reduction = 500
        print("Building feature matrix")

        print("\tBuilding Part Of Speech sequence matrix")
        pos_sequence_matrix = np.stack(Parallel(n_jobs=cpu_count(), verbose=0)(
            delayed(build_sentence_pos_sequences)(text_series[index], pos_indices) for index in
            range(len(text_series))))

        print("\tBuilding Word sequence matrix")
        word_sequence_matrix = np.stack([build_sentence_word_sequences(
            text_series[index], word_indices) for index in range(len(text_series))])

        print("\tBuilding Char sequence matrix")
        char_sequence_matrix = np.stack(Parallel(n_jobs=cpu_count(), verbose=0)(
            delayed(build_sentence_char_sequences)(text_series[index], char_indices) for index in
            range(len(text_series))))

        print("\tGenerating additional features of documents")
        feature_matrix = DataFrame(index=list(range(len(text_series))), columns=additional_feature_list)
        feature_matrix = DataFrame(Parallel(n_jobs=cpu_count(), verbose=0)(
            delayed(get_additional_features)(text_series[index], row) for index, row in feature_matrix.iterrows()))

        # print("\tGenerating rare POS bi-grams with min_df =", min_f, "and max_df =", max_f)
        # pos_text_list = Parallel(n_jobs=cpu_count(), verbose=0)(
        #     delayed(convert_text_to_pos_tags)(text_series[index]) for index in range(len(text_series)))
        # rare_pos_bi_grams = TfidfVectorizer(ngram_range=(2, 2),
        #                                     min_df=min_f, max_df=max_f).fit_transform(pos_text_list)
        # print("\t\tFeatures generated:", rare_pos_bi_grams.shape[1])
        # rare_pos_bi_grams = rare_pos_bi_grams.toarray()
        #
        # min_f = 0.001
        # print("\tGenerating common spelling error uni-grams with min_df =", min_f, "and max_df =", max_f)
        # spelling_errors = DataFrame(Parallel(n_jobs=cpu_count(), verbose=0)(
        #     delayed(get_spelling_errors)(text_series[index]) for index in range(len(text_series))),
        #     index=list(range(len(text_series))))
        # common_spelling_errors = TfidfVectorizer(ngram_range=(1, 1), min_df=min_f, max_df=max_f,
        #                                          lowercase=False).fit_transform(spelling_errors[0])
        # print("\t\tFeatures generated:", common_spelling_errors.shape[1])
        # common_spelling_errors = common_spelling_errors.toarray()

        print("\tGenerating word uni-grams for entire corpus")
        word_unigrams = TfidfVectorizer().fit_transform(text_series)
        if word_unigrams.shape[1] > reduction:
             print("\tApplying SVD to reduce feature size from", word_unigrams.shape[1], "to", reduction)
             word_unigrams = TruncatedSVD(n_components=reduction).fit_transform(word_unigrams)
        else:
            word_unigrams = word_unigrams.toarray()
        feature_matrix = DataFrame(np.hstack((feature_matrix.as_matrix(),
                                              # rare_pos_bi_grams,
                                              # common_spelling_errors,
                                              word_unigrams)))

        print("\tSaving feature matrices to", root_path + "model/")
        with open(root_path + "model/" + args.data_file + ".additional_features.pickle", "wb") as file_to_write:
            pickle.dump(feature_matrix, file_to_write, protocol=4)
        with open(root_path + "model/" + args.data_file + ".pos_sequence_matrix.pickle", "wb") as file_to_write:
            pickle.dump(pos_sequence_matrix, file_to_write, protocol=4)
        with open(root_path + "model/" + args.data_file + ".word_sequence_matrix.pickle", "wb") as file_to_write:
            pickle.dump(word_sequence_matrix, file_to_write, protocol=4)
        with open(root_path + "model/" + args.data_file + ".char_sequence_matrix.pickle", "wb") as file_to_write:
            pickle.dump(char_sequence_matrix, file_to_write, protocol=4)

    return feature_matrix, pos_sequence_matrix, word_sequence_matrix, char_sequence_matrix


def convert_text_to_pos_tags(doc):
    return " ".join([tag for word, tag in pos_tag(doc.split())]).strip()


def build_sentence_pos_sequences(text, pos_indices):
    doc_matrix = np.zeros((max_num_of_sentences, max_num_of_tokens_per_sentence), dtype=np.int64)
    for i, sentence in enumerate(text.split('\n')):
        if i < max_num_of_sentences:
            for j, pos in enumerate(pos_tag(sentence.split()[:max_num_of_tokens_per_sentence])):
                doc_matrix[i, j] = pos_indices[pos[1]]
    return doc_matrix


def build_sentence_word_sequences(text, word_indices):
    doc_matrix = np.zeros((max_num_of_sentences, max_num_of_tokens_per_sentence), dtype=np.int64)
    for i, sentence in enumerate(text.split('\n')):
        if i < max_num_of_sentences:
            for j, word in enumerate(sentence.lower().split()[:max_num_of_tokens_per_sentence]):
                doc_matrix[i, j] = word_indices[word]
    return doc_matrix


def build_sentence_char_sequences(text, char_indices):
    doc_matrix = np.zeros((max_num_of_sentences, max_num_of_chars), dtype=np.int64)

    for i, sentence in enumerate(text.split('\n')):
        if i < max_num_of_sentences:
            for j, char in enumerate(sentence[:max_num_of_chars]):
                doc_matrix[i, j] = char_indices[char]
    return doc_matrix


def get_pos_tag_sequence(text):
    sentences = text.split('\n')
    list_of_pos_sentences = list()
    for sent in sentences:
        word_tokens = sent.split()
        list_of_pos_sentences.append([tag for word, tag in pos_tag(word_tokens)])
    return list_of_pos_sentences


def get_additional_features(text, df_row):
    sentences = text.split('\n')
    tokens = text.split()

    num_of_tokens = len(tokens)
    # avg_word_len = np.mean([len(word) for word, token in pos_tag(tokens) if token not in punctuation])
    det_token_ratio = Counter(token for _, token in pos_tag(tokens))['DT'] / len(tokens)
    punct_token_ratio = sum([1 for _, tag in pos_tag(tokens) if tag in punctuation]) / len(tokens)
    num_of_long_sentences = sum([1 for sentence in sentences if len(sentence.split()) > 60])
    num_of_short_sentences = sum([1 for sentence in sentences if len(sentence.split()) < 5])
    # num_of_passive_sentences = sum([1 for sentence in sentences if is_passive_sentence(sentence.split())])

    df_row[additional_feature_list] = [num_of_tokens,
                                       # avg_word_len,
                                       det_token_ratio,
                                       punct_token_ratio,
                                       num_of_long_sentences,
                                       num_of_short_sentences]  # num_of_passive_sentences]

    return df_row


def is_passive_sentence(sentence):
    if 'by' in sentence:
        location = sentence.index('by')
        if location > 1 and 'VB' == pos_tag(sentence)[location - 1][1]:
            return True
    return False


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


def get_feature_metrics(feature_matrix):
    import matplotlib.pyplot as plt

    objects = feature_matrix['label'].unique()
    y_pos = np.arange(len(objects))
    features = ['Average number of tokens',
                'Average determiner to token ratio',
                'Average punctuation to token ratio',
                'Average number of long sentences',
                'Average number of short sentences',
                'Average number of passive sentences']
    print("", *objects, sep='\t')
    for feature in range(len(additional_feature_list)):
        mean = [feature_matrix.loc[feature_matrix['label'] == label, feature].mean() for label in objects]

        print(features[feature], *mean, sep='\t')
        plt.bar(y_pos, mean, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title(features[feature])

        plt.show()

    return


def get_total_sentence_count(text_series):
    return sum([len(text_series[i].split('\n')) for i in range(text_series.shape[0])])


def plot_text_lengths(doc_text):
    import matplotlib.pyplot as plt
    import seaborn as sns

    list_of_pos_tags = Parallel(n_jobs=cpu_count())(delayed(get_pos_tag_sequence)(text) for text in doc_text)
    sent_lengths = [len(doc) for doc in list_of_pos_tags]
    token_lengths = [len(sent) for doc in list_of_pos_tags for sent in doc]
    char_lengths = [len(sent) for doc in doc_text for sent in doc.split('\n')]

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

    plt.title("Number of characters per sentence")
    plt.xlabel("Average number of characters per sentence: " + str(
        np.mean(char_lengths)) + "\nLongest number of characters in one sentence: " + str(max(char_lengths)))
    plt.ylabel(str(int(
        len([len(sent) for doc in doc_text for sent in doc.split('\n') if len(sent) < 250]) / len(
            char_lengths) * 100)) + "% of sentences have under 250 characters")
    plt.axis([0, 500, 0, 1])
    sns.boxplot(char_lengths)
    plt.show()


def find_word_n_grams(word_list, n=2):
    return zip(*[word_list[i:] for i in range(n)])


def find_char_n_grams(string, n=2):
    string = sub(r'\s+', '', string)
    return zip(*[string[i:] for i in range(n)])



def train_val_test_split(df):

    x_train, x_test, y_train, y_test = train_test_split(df, df['Language'], stratify=df[['Language']], #, 'Prompt'
                                                        test_size=0.1, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, x_test['Language'], stratify=x_test[['Language']], #, 'Prompt'
                                                    test_size=0.1, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test


def main():
    df = read_data(args.data_file)
    # df = df[df.Language != 'Hindi']
    # df = df[df.Language != 'Telugu']
    # df.drop(columns='Prompt', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    labels_series = df['Language']
    text_series = df['Text']
    unique_labels_count = len(Counter(labels_series))

    word_model, pos_model, char_model = get_word2vec_model(text_series)
    word_indices = dict((p, i) for i, p in enumerate(word_model.wv.vocab, start=1))
    pos_indices = dict((p, i) for i, p in enumerate(pos_model.wv.vocab, start=1))
    char_indices = dict((p, i) for i, p in enumerate(char_model.wv.vocab, start=1))

    feature_matrix, pos_sequence_matrix, word_sequence_matrix, char_sequence_matrix = get_feature_matrix(text_series,
                                                                                                         word_indices,
                                                                                                         pos_indices,
                                                                                                         char_indices)
    #feature_matrix = get_feature_matrix(text_series)

    print("Feature matrix shape:", feature_matrix.shape[1])
    encoder = LabelEncoder().fit(labels_series)
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    # history = LossHistory()

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(df)

    one_hot_train_labels = to_categorical(encoder.transform(y_train), num_classes=unique_labels_count)
    one_hot_val_labels = to_categorical(encoder.transform(y_val), num_classes=unique_labels_count)

    train_pos_input = pos_sequence_matrix[x_train.index]
    train_word_input = word_sequence_matrix[x_train.index]
    train_char_input = char_sequence_matrix[x_train.index]

    val_pos_input = pos_sequence_matrix[x_val.index]
    val_word_input = word_sequence_matrix[x_val.index]
    val_char_input = char_sequence_matrix[x_val.index]

    epoch_num = 50
    print("Training neural network for", epoch_num, "epoch(s)")
    if args.sequence == "pos":
        if args.neural_network == "lstm":
            nn_model, final_layer = build_pos_sequence_lstm_model(pos_indices, pos_model, unique_labels_count)
        else:
            nn_model, final_layer = build_pos_sequence_conv_model(unique_labels_count)
        nn_model.summary()
        nn_model.fit(train_pos_input, one_hot_train_labels,
                     validation_data=(val_pos_input, one_hot_val_labels), epochs=epoch_num,
                     verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        evaluation = nn_model.evaluate(val_pos_input, one_hot_val_labels, verbose=0)
        nn_predictions = nn_model.predict(pos_sequence_matrix, verbose=0)
    elif args.sequence == "word":
        if args.neural_network == "lstm":
            nn_model, final_layer = build_word_sequence_lstm_model(word_indices, word_model, unique_labels_count)
        else:
            nn_model, final_layer = build_word_sequence_conv_model(word_indices, word_model, unique_labels_count)
        nn_model.summary()
        nn_model.fit(train_word_input, one_hot_train_labels,
                     validation_data=(val_word_input, one_hot_val_labels), epochs=epoch_num,
                     verbose=2, batch_size=32, shuffle=True, callbacks=[earlystop_cb, check_cb])
        evaluation = final_layer.evaluate(val_word_input, one_hot_val_labels, verbose=0)
        nn_predictions = final_layer.predict(word_sequence_matrix, verbose=0)
    elif args.sequence == "char":
        if args.neural_network == "lstm":
            nn_model, final_layer = build_char_sequence_lstm_model(char_indices, char_model, unique_labels_count)
        else:
            nn_model, final_layer = build_char_sequence_conv_model(unique_labels_count)
        nn_model.summary()
        nn_model.fit(train_char_input, one_hot_train_labels,
                     validation_data=(val_char_input, one_hot_val_labels), epochs=epoch_num,
                     verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        evaluation = nn_model.evaluate(val_char_input, one_hot_val_labels, verbose=0)
        nn_predictions = nn_model.predict(char_sequence_matrix, verbose=0)
    elif args.sequence == "combined":
        if args.neural_network == "lstm":
            nn_model, final_layer = build_combined_lstm_model(word_indices, pos_indices, char_indices, word_model,
                                                              pos_model, char_model, unique_labels_count)
        else:
            nn_model, final_layer = build_combined_conv_model(word_indices, word_model, unique_labels_count)
        nn_model.summary()
        nn_model.fit([train_pos_input, train_word_input, train_char_input], one_hot_train_labels,
                     validation_data=([val_pos_input, val_word_input, val_char_input], one_hot_val_labels),
                     epochs=epoch_num, verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        evaluation = nn_model.evaluate([val_pos_input, val_word_input, val_char_input], one_hot_val_labels)
        nn_predictions = nn_model.predict([pos_sequence_matrix, word_sequence_matrix, char_sequence_matrix])
    elif args.sequence == "combined_2":
        if args.neural_network == "lstm":
            pos_model, pos_final_layer = build_pos_sequence_lstm_model(pos_indices, pos_model, unique_labels_count)
            word_model, word_final_layer = build_word_sequence_lstm_model(word_indices, word_model, unique_labels_count)
            char_model, char_final_layer = build_char_sequence_lstm_model(char_indices, char_model, unique_labels_count)
        else:
            pos_model, pos_final_layer = build_pos_sequence_conv_model(unique_labels_count)
            word_model, word_final_layer = build_word_sequence_conv_model(word_indices,
                                                                          word_model,
                                                                          unique_labels_count)
            char_model, char_final_layer = build_char_sequence_conv_model(unique_labels_count)
        print("POS Model:")
        pos_model.summary()
        print("Word Model:")
        word_model.summary()
        print("Char Model:")
        char_model.summary()
        pos_model.fit(train_pos_input, one_hot_train_labels,
                      validation_data=(val_pos_input, one_hot_val_labels), epochs=epoch_num,
                      verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        pos_evaluation = pos_model.evaluate(val_pos_input, one_hot_val_labels, verbose=0)
        pos_predictions = pos_model.predict(pos_sequence_matrix, verbose=0)
        word_model.fit(train_word_input, one_hot_train_labels,
                       validation_data=(val_word_input, one_hot_val_labels), epochs=epoch_num,
                       verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        word_evaluation = word_model.evaluate(val_word_input, one_hot_val_labels, verbose=0)
        word_predictions = word_model.predict(word_sequence_matrix, verbose=0)
        char_model.fit(train_char_input, one_hot_train_labels,
                       validation_data=(val_char_input, one_hot_val_labels), epochs=epoch_num,
                       verbose=2, batch_size=50, shuffle=True, callbacks=[earlystop_cb, check_cb])
        char_evaluation = char_model.evaluate(val_char_input, one_hot_val_labels, verbose=0)
        char_predictions = char_model.predict(char_sequence_matrix, verbose=0)
        nn_predictions = np.hstack((pos_predictions,
                                    word_predictions,
                                    char_predictions))
        evaluation = np.mean((pos_evaluation[0], word_evaluation[0], char_evaluation[0])
                             ), np.mean((pos_evaluation[1], word_evaluation[1], char_evaluation[1]))
    else:
        nn_model = build_resnet_model(word_indices, word_model, unique_labels_count)
        nn_model.summary()
        nn_model.fit(train_word_input, one_hot_train_labels,
                     validation_data=(val_word_input, one_hot_val_labels), epochs=epoch_num,
                     verbose=2, batch_size=32, shuffle=True, callbacks=[earlystop_cb, check_cb])
        nn_predictions = nn_model.predict(word_sequence_matrix, verbose=0)
        evaluation = nn_model.evaluate(val_word_input, one_hot_val_labels, verbose=0)

    #feature_matrix_2 = concat([feature_matrix, DataFrame(nn_predictions)], axis=1)
    feature_matrix_2 =feature_matrix

    print("Matrix shape:", feature_matrix_2.shape)
    #clf =RandomForestClassifier(n_estimators=100,criterion='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 50, min_weight_fraction_leaf = 0.0, max_features ='auto', max_leaf_nodes = 70, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, oob_score = False, n_jobs = 4, random_state = None, verbose = 0, warm_start = False, class_weight = None)
    clf = LinearSVC(dual=False)
    print("Classifiying with:\n", clf, sep='\t')


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
    main()
