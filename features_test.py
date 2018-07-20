#! /usr/bin/env python3
import itertools
from collections import Counter
from multiprocessing import cpu_count
from string import punctuation

import numpy as np
from enchant.checker import SpellChecker
from joblib import Parallel, delayed
from nltk import pos_tag
from pandas import read_table, DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

additional_feature_list = ['num_of_tokens',
                           'avg_word_len',
                           'det_token_ratio',
                           'punct_token_ratio',
                           'num_of_long_sentences',
                           'num_of_short_sentences',
                           'num_of_passive_sentences']


def read_data(file='toefl11_tokenized.tsv'):
    print("Reading index from", file)
    return read_table(file)


def get_additional_features(text, df_row):
    sentences = text.split('\n')
    tokens = text.split()

    num_of_tokens = len(tokens)
    avg_word_len = np.mean([len(word) for word, token in pos_tag(tokens) if token not in punctuation])
    det_token_ratio = Counter(token for _, token in pos_tag(tokens))['DT'] / len(tokens)
    punct_token_ratio = sum([1 for _, tag in pos_tag(tokens) if tag in punctuation]) / len(tokens)
    num_of_long_sentences = sum([1 for sentence in sentences if len(sentence.split()) > 60])
    num_of_short_sentences = sum([1 for sentence in sentences if len(sentence.split()) < 5])
    num_of_passive_sentences = sum([1 for sentence in sentences if is_passive_sentence(sentence.split())])

    df_row[additional_feature_list] = [num_of_tokens,
                                       avg_word_len,
                                       det_token_ratio,
                                       punct_token_ratio,
                                       num_of_long_sentences,
                                       num_of_short_sentences,
                                       num_of_passive_sentences]

    return df_row


def is_passive_sentence(sentence):
    if 'by' in sentence:
        location = sentence.index('by')
        if location > 1 and 'VB' == pos_tag(sentence)[location - 1][1]:
            return True
    return False


def convert_text_to_pos_tags(doc):
    return " ".join([tag for word, tag in pos_tag(doc.split())]).strip()


def get_spelling_errors(text):
    return " ".join([err.word for err in SpellChecker("en_US", text)]) or ""


def main():
    df = read_data()
    text_series = df['Text']
    min_f = 0.0005
    max_f = 0.20
    reduction = 500

    print("\tGenerating additional features of documents")
    feature_matrix = DataFrame(index=list(range(len(text_series))), columns=additional_feature_list)
    feature_matrix = DataFrame(Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(get_additional_features)(text_series[index], row) for index, row in feature_matrix.iterrows()))

    print("\tGenerating rare POS bi-grams with min_df =", min_f, "and max_df =", max_f)
    pos_text_list = Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(convert_text_to_pos_tags)(text_series[index]) for index in range(len(text_series)))
    rare_pos_bi_grams = TfidfVectorizer(ngram_range=(2, 2), min_df=min_f, max_df=max_f).fit_transform(pos_text_list)
    print("\t\tFeatures generated:", rare_pos_bi_grams.shape[1])
    rare_pos_bi_grams = rare_pos_bi_grams.toarray()

    min_f = 0.001
    print("\tGenerating common spelling error uni-grams with min_df =", min_f, "and max_df =", max_f)
    spelling_errors = DataFrame(Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(get_spelling_errors)(text_series[index]) for index in range(len(text_series))),
        index=list(range(len(text_series))))
    common_spelling_errors = TfidfVectorizer(ngram_range=(1, 1), min_df=min_f, max_df=max_f,
                                             lowercase=False).fit_transform(spelling_errors[0])
    print("\t\tFeatures generated:", common_spelling_errors.shape[1])
    common_spelling_errors = common_spelling_errors.toarray()

    print("\tGenerating word uni-grams for entire corpus")
    word_unigrams = TfidfVectorizer().fit_transform(text_series)
    if word_unigrams.shape[1] > reduction:
        print("\tApplying SVD to reduce feature size from", word_unigrams.shape[1], "to", reduction)
        word_unigrams = TruncatedSVD(n_components=reduction).fit_transform(word_unigrams)
    else:
        word_unigrams = word_unigrams.toarray()

    # combinations = ((), rare_pos_bi_grams, common_spelling_errors, word_unigrams,
    #                 np.hstack((rare_pos_bi_grams, common_spelling_errors)),
    #                 np.hstack((rare_pos_bi_grams, word_unigrams)),
    #                 np.hstack((common_spelling_errors, word_unigrams)),
    #                 np.hstack((rare_pos_bi_grams, common_spelling_errors, word_unigrams)))
    # combinations = (word_unigrams,
    #                 np.hstack((rare_pos_bi_grams, word_unigrams)),
    #                 np.hstack((common_spelling_errors, word_unigrams)),
    #                 np.hstack((rare_pos_bi_grams, common_spelling_errors, word_unigrams)))
    combinations = [word_unigrams]
    best_macro = 0
    best_subset = set()
    best_j = 0
    for i in range(feature_matrix.shape[1]):
        for subset in itertools.combinations(feature_matrix, i + 1):
            matrix = feature_matrix[list(subset)]
            for j in range(len(combinations)):
                # if j == 0:
                #     matrix2 = matrix
                # else:
                matrix2 = np.hstack((matrix.as_matrix(), combinations[j]))
                print("testing with subset:", subset, "with combination", j)
                x_train, x_test, y_train, y_test = train_test_split(matrix2, df['Language'],
                                                                    test_size=0.09, random_state=0)
                clf = LinearSVC(dual=False)
                clf.fit(x_train, y_train)
                predictions = clf.predict(x_test)
                macro = f1_score(y_test, predictions, average='macro')
                print("macro:", macro)
                if macro > best_macro:
                    best_macro = macro
                    best_subset = subset
                    best_j = j
                    print("New best macro:", best_macro)
    print("Best macro:", best_macro)
    print("Best subset:", best_subset, "with j =", best_j)


if __name__ == '__main__':
    print("Detected", cpu_count(), "CPU cores")
    # main()
