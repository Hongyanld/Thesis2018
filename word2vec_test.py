#! /usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from multiprocessing import cpu_count

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from joblib import Parallel, delayed
from keras import Sequential, callbacks
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from nltk import pos_tag
from pandas import read_table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

parser = ArgumentParser()
parser.add_argument('-e', '--epochs', metavar='INT', type=int, default=10)
parser.add_argument('type', metavar='TYPE', type=str, default="word", help='TYPE can be one of the following: "pos", "word", "char"')
args = parser.parse_args()


def read_data(file='toefl11_tokenized.tsv'):
    print("Reading index from", file)
    return read_table(file)


def train_val_test_split(df):

    x_train, x_test, y_train, y_test = train_test_split(df, df['Language'], stratify=df[['Language', 'Prompt']],
                                                        test_size=2200, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, x_test['Language'], stratify=x_test[['Language', 'Prompt']],
                                                    test_size=1100, random_state=0)
    return x_train, x_val, y_train, y_val


def convert_text_to_pos_tags(doc):
    return " ".join([tag for word, tag in pos_tag(doc.split())]).strip()


def build_tagged_word_document(doc, doc_id):
    words = [word for word in doc.split()]
    return TaggedDocument(words, [doc_id])


def build_tagged_char_document(doc, doc_id):
    chars = [char for char in doc]
    return TaggedDocument(chars, [doc_id])


def build_tagged_pos_document(doc, doc_id):
    pos = [pos for pos in convert_text_to_pos_tags(doc).split()]
    return TaggedDocument(pos, [doc_id])


def build_tagged_documents(df, token_type):
    if token_type == 'word':
        return Parallel(n_jobs=cpu_count())(
            delayed(build_tagged_word_document)(paper, paper_id) for paper, paper_id in zip(df['Text'], df['Filename']))
    elif token_type == 'char':
        return Parallel(n_jobs=cpu_count())(
            delayed(build_tagged_char_document)(paper, paper_id) for paper, paper_id in zip(df['Text'], df['Filename']))
    elif token_type == 'pos':
        return Parallel(n_jobs=cpu_count())(
            delayed(build_tagged_pos_document)(paper, paper_id) for paper, paper_id in zip(df['Text'], df['Filename']))
    else:
        exit(1)


def neural_net_test(token_type):
    df = read_data()

    data = build_tagged_documents(df, token_type)
    doc_model = Doc2Vec(data, dm=0, vector_size=300, alpha=0.02,
                        window=1, min_count=1, workers=1, epochs=70)
    feature_matrix = np.ndarray((df.shape[0], 300))

    for i in range(df.shape[0]):
        feature_matrix[i] = doc_model[df.loc[i, 'Filename']]

    encoder = LabelEncoder().fit(df['Language'])
    file_name = os.path.basename(sys.argv[0]).split('.')[0]
    check_cb = callbacks.ModelCheckpoint('checkpoints/' + str(file_name) + str(os.getpid()) + '.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    x_train, x_test, y_train, y_test = train_val_test_split(df)

    one_hot_train_labels = to_categorical(encoder.transform(y_train), num_classes=11)
    one_hot_test_labels = to_categorical(encoder.transform(y_test), num_classes=11)

    model = Sequential()
    model.add(Dense(512, input_shape=(300,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(11))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(feature_matrix[x_train.index], one_hot_train_labels,
              validation_data=(feature_matrix[x_test.index], one_hot_test_labels), epochs=30, verbose=2, batch_size=32,
              shuffle=True, callbacks=[earlystop_cb, check_cb])
    print(model.evaluate(feature_matrix[x_test.index], one_hot_test_labels, verbose=0))
    nn_predictions = model.predict(feature_matrix)
    clf = LinearSVC(dual=False)
    clf.fit(nn_predictions[x_train.index], df.loc[y_train.index, 'Language'])
    predictions = clf.predict(nn_predictions[x_test.index])
    print(classification_report(y_test, predictions))


def gird_test_embeddings(epoch_grid, token_type):
    from itertools import product
    df = read_data()

    print("\tBuilding lists of tokens")
    data = build_tagged_documents(df, token_type)

    dm_grid = [0, 1]
    alpha_grid = [0.005*i for i in range(1, 11, 1)]
    window_grid = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    word_embeddings_sizes = [100, 128, 200, 256, 300]
    print(epoch_grid, dm_grid, alpha_grid, window_grid, word_embeddings_sizes, sep='\n')

    configs = Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(do_word2vec_test)(df, data, epoch, dm, alpha, window, size) for epoch, dm, alpha, window, size in
        product(epoch_grid, dm_grid, alpha_grid, window_grid, word_embeddings_sizes))
    best = max(configs, key=lambda x: x[0])
    print("Best macro for", args.type, ":", best[0])
    print("Best config for", args.type, ": epochs=%d, dm=%d, alpha=%f, window=%d, size=%d" % (best[1]))


def do_word2vec_test(df, data, epochs, dm, alpha, window, size):
    doc_model = Doc2Vec(data, dm=dm, vector_size=size, alpha=alpha,
                        window=window, min_count=1, workers=1, epochs=epochs)
    feature_matrix = np.ndarray((df.shape[0], size))

    for i in range(df.shape[0]):
        feature_matrix[i] = doc_model[df.loc[i, 'Filename']]
    x_train, x_test, y_train, y_test = train_val_test_split(df)
    clf = LinearSVC(dual=False)
    clf.fit(feature_matrix[x_train.index], y_train)
    predictions = clf.predict(feature_matrix[x_test.index])
    macro = f1_score(y_test, predictions, average='macro')

    return macro, (epochs, dm, alpha, window, size)


def do_bag_of_words_test():
    df = read_data()

    feature_matrix = TfidfVectorizer().fit_transform(df['Text'])
    print(feature_matrix.shape)
    x_train, x_test, y_train, y_test = train_val_test_split(df)
    clf = LinearSVC(dual=False)
    clf.fit(feature_matrix[x_train.index], y_train)
    predictions = clf.predict(feature_matrix[x_test.index])
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    print(args)
    print("Detected", cpu_count(), "CPU cores")
    # gird_test_embeddings([args.epochs], args.type)
