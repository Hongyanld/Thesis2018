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


#hybao with SVM veridition

import os
import pickle
import sys
import operator
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
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from model import *

from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument('-m', '--model', action='store_true')
parser.add_argument('-f', '--feature_matrix', action='store_true')
parser.add_argument('-k', '--k_value', metavar='INT', type=int, default=5)
parser.add_argument('-d', '--data_file', metavar='PATH', type=str, default='toefl11_tokenized.tsv')
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

def get_feature_matrix(text_series):
    if args.feature_matrix:
        print("Loading feature matrices from", root_path + "model/")
        with open(root_path + "model/" + args.data_file + ".additional_features.pickle", "rb") as file_to_read:
            feature_matrix = pickle.load(file_to_read)

    else:
        # min_f = 0.0005
        # max_f = 0.20
        reduction = 500
        print("Building feature matrix")

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
        word_unigrams = CountVectorizer(text_series)
        word_unigrams = TfidfVectorizer().fit_transform(text_series)
        word_unigrams = TfidfVectorizer().get_stop_words()
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

    return feature_matrix

# def Label_matrix(feature_matrix):
#     matrix = feature_matrix.join(df["Language"])
#     df_group = matrix.groupby("Language")
#     label_matrix = df_group.sum()
#
#     matrix.loc[matrix["Language"] == "Chinese"]
#     with open(root_path + "model/" + args.data_file + ".label_features.pickle", "wb") as file_to_write:
#         pickle.dump(label_matrix, file_to_write, protocol=4)
# return label_matrix
def demean(arr):
    return arr - arr.mean()


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

def main():
    df = read_data(args.data_file)
    # df = df[df.Language != 'Hindi']
    # df = df[df.Language != 'Telugu']
    # df.drop(columns='Prompt', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    labels_series = df['Language']
    text_series = df['Text']
    unique_labels_count = len(Counter(labels_series))



    feature_matrix, pos_sequence_matrix, word_sequence_matrix, char_sequence_matrix = get_feature_matrix(text_series,
                                                                                                         word_indices,
                                                                                                         pos_indices,
                                                                                                         char_indices)
    #feature_matrix = get_feature_matrix(text_series)
    matrix = feature_matrix.join(df["Language"])

    print("Feature matrix shape:", feature_matrix.shape[1])

    encoder = LabelEncoder().fit(labels_series)
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    # history = LossHistory()

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(matrix)

    x_in = x_train.append(x_val).groupby("Language").transform(demean)
    # 从各组中减去平均值。即距平化函数处理。我们先写出这个函数,然后应用这个函数到各个分组

    y_out = labels_series[x_train.index.append(x_val.index)] #


    print("Matrix shape:", feature_matrix.shape)
    #clf = RandomForestClassifier(n_estimators=100,criterion='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 50, min_weight_fraction_leaf = 0.0, max_features ='auto', max_leaf_nodes = 70, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, oob_score = False, n_jobs = 4, random_state = None, verbose = 0, warm_start = False, class_weight = None)
    clf= LinearSVC(dual=False)
    print("Classifiying with:\n", clf, sep='\t')

    print("Training Classifier")
    start = time()
    clf.fit(x_in, y_out)
    print("Trained classifier in", time() - start, "seconds")

    print("Predicting test set")
    predictions = clf.predict(feature_matrix.loc[x_test.index])
    print(classification_report(y_test, predictions))
    print("Accuracy score:", accuracy_score(y_test, predictions))
    print("Micro F1:", f1_score(y_test, predictions, average='micro'))
    print("Macro F1:", f1_score(y_test, predictions, average='macro'))
    print("Neural Network Loss:", evaluation[0])
    print("Neural Network Accuracy:", evaluation[1])


if __name__ == '__main_1__':
    print(args)
    print("Detected", cpu_count(), "CPU cores")
    #main_1()
