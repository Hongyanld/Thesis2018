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

from multiprocessing import cpu_count

from joblib import Parallel, delayed
from pandas import DataFrame, read_csv, concat

root_path = "/Users/baoh/PycharmProjects/Thesis2018/ETS_Corpus_of_Non-Native_Written_English/data/text/"


def read_index_csv():
    print("Reading index from", root_path + "index.csv")
    raw_df = read_csv(root_path + "index.csv")
    raw_df.drop(columns=['Score Level'], inplace=True)
    raw_df['Prompt'].replace({'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5, 'P6': 6, 'P7': 7, 'P8': 8}, inplace=True)
    raw_df['Language'].replace({'ARA': 'Arabic', 'DEU': 'German', 'FRA': 'French',
                                'HIN': 'Hindi', 'ITA': 'Italian', 'JPN': 'Japanese',
                                'KOR': 'Korean', 'SPA': 'Spanish', 'TEL': 'Telugu',
                                'TUR': 'Turkish', 'ZHO': 'Chinese'}, inplace=True)
    raw_df['Text'] = None
    return raw_df


def get_original_text(row):
    with open(root_path + "responses/original/" + row['Filename'], "r") as f:
        row['Text'] = f.read()
    return row


def get_tokenized_sentences(row, split=True):
    with open(root_path + "responses/tokenized/" + row['Filename'], "r") as f:
        if split:
            row['Text'] = list(filter(None, f.read().split('\n')))
        else:
            row['Text'] = f.read()
    return row


def split_text_into_sentences(row):
    new_df = DataFrame(columns=['Filename', 'Language', 'Sentence', 'Text'])
    list_of_sents = row['Text']
    for index in range(len(list_of_sents)):
        new_df.loc[index] = [row['Filename'], row['Language'], str(index), list_of_sents[index]]
    return new_df


def main():
    raw_df = read_index_csv()

    raw_df = DataFrame(
        Parallel(n_jobs=cpu_count())(delayed(get_original_text)(row) for _, row in raw_df.iterrows()))
    tokenized_df = DataFrame(
        Parallel(n_jobs=cpu_count())(delayed(get_tokenized_sentences)(row, True) for _, row in raw_df.iterrows()))
    tokenized_df2 = DataFrame(
        Parallel(n_jobs=cpu_count())(delayed(get_tokenized_sentences)(row, False) for _, row in raw_df.iterrows()))
    tokenized_df = concat(Parallel(n_jobs=cpu_count(), verbose=0)(
        delayed(split_text_into_sentences)(row) for _, row in tokenized_df.iterrows()), ignore_index=True)

    print("Writing data to tsv file")
    raw_df.to_csv("toefl11_original.tsv", sep='\t', index=False)
    tokenized_df.to_csv("toefl11_tokenized_split.tsv", sep='\t', index=False)
    tokenized_df2.to_csv("toefl11_tokenized.tsv", sep='\t', index=False)


if __name__ == '__main__':
    print("Detected", cpu_count(), "CPU cores")
    main()
