# find several ways to transform the word, term ,doc2 vector
# bag-of-ngrams
word2vec
Word Embedding和Sentence/Document EMbedding
CBOW”和“Skip-gram
Word2vec其实是N-gram模型训练的副产品



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



def read_data(file):
    print("Reading index from", file)
    return read_table(file)

def get_word2vec_model(text_series):
    if args.model:
        print("Loading word2vec models from", root_path + "model_vector/")
        word_model = Word2Vec.load(root_path + "model_vector/toefl11.word_model.bin")
    else:
        print("Training word2vec model")
        word_model = train_word2vec_model(text_series)

    return word_model


def train_word2vec_model(text_series):
    print("\tBuilding lists of tokens")
    word_data = [sent.lower().split() for doc in text_series for sent in doc.split('\n') if sent]

    epochs = 70
    print("\tTraining word2vec word model with", epochs, "epochs")
    start = time()
    word_model = Word2Vec(word_data, size=word_embeddings_size, alpha=0.02, window=1, min_count=1,
                          workers=cpu_count(), sg=1, iter=epochs)
    print("\tTrained word2vec word model in", (time() - start) / 60, "minutes")



    print("\tSaving word2vec models to", root_path + "model_vector/")
    word_model.save(root_path + "model_vector/toefl11.word_model.bin")

    return word_model