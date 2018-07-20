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

parser = ArgumentParser()
parser.add_argument('-e', '--epochs', metavar='INT', type=int, default=10)
parser.add_argument('type', metavar='TYPE', type=str, default='word', help='TYPE can be one of the following: "pos", "word", "char"')
args = parser.parse_args()


root_path = "./"
word_embeddings_size = 300

def read_data(file='toefl11_tokenized.tsv'):
    print("Reading index from", file)
    return read_table(file)


 print("Loading word2vec models from", root_path + "model/")
        word_model = Word2Vec.load(root_path + "model/toefl11.word_model.bin")
