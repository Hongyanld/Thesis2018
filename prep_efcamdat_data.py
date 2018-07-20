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

import xml.etree.ElementTree as eTree
from multiprocessing import cpu_count
from re import sub
from string import punctuation
from time import time

from joblib import Parallel, delayed
from nltk import sent_tokenize, word_tokenize, pos_tag
from pandas import DataFrame

xml_file = "EF201403_selection601.xml"

print("Creating tree for", xml_file)
with open(xml_file, 'r') as f:
    xml_string = f.read()

xml_string = sub(r'<br>', '', xml_string)
xml_string = sub(r'(<br/>)+', ' ', xml_string)
tree = eTree.ElementTree(eTree.fromstring(xml_string))
root = tree.getroot()


def parse_xml(index):
    return [root[1][index].attrib['id'],
            root[1][index][0].attrib['nationality'],
            "\n".join(" ".join(word_tokenize(sent)) for sent in sent_tokenize(root[1][index][4].text.strip()))]


def convert_words_into_pos(row):
    """
    Converts words in row['text'] into a string of nltk POS tags

    :param row: pandas Dataframe row
    :return: The modified pandas Dataframe row
    """
    row['text'] = " ".join([tag for word, tag in pos_tag(word_tokenize(row['text']))]).strip()
    return row


def split_text_into_sentences(row):
    """
    Takes a pandas Dataframe row and creates a new Dataframe with rows for each sentence in row['text']

    :param row: pandas Dataframe row
    :return: A new Dataframe containing row['text'] split into sentences
    """
    new_df = DataFrame(columns=['id', 'label', 'text'])
    list_of_sents = sent_tokenize(row['text'])
    for index in range(len(list_of_sents)):
        len_of_sent = sum([1 for word in word_tokenize(list_of_sents[index]) if word not in punctuation])
        if len_of_sent > 2:
            new_df.loc[index] = [row['id'] + "_" + str(index), row['label'], list_of_sents[index]]
        else:
            new_df.loc[index] = None
    return new_df


def convert_data_into_pos_and_sentences(row):
    """
    Combines converting words to POS tags and splitting the documents into sentences
    :param row: pandas Dataframe row
    :return: A new Dataframe containing row['text'] split into sentences and converted into POS tags
    """
    new_df = DataFrame(columns=['id', 'label', 'text'])
    row['text'] = " ".join([tag for word, tag in pos_tag(word_tokenize(row['text']))]).strip()
    list_of_sents = sent_tokenize(row['text'])
    for index in range(len(list_of_sents)):
        new_df.loc[index] = [row['id'] + "_" + str(index), row['label'], list_of_sents[index]]
    return new_df


def main():
    num_of_writings = len(root.findall('.//writing'))
    df = DataFrame(index=range(num_of_writings), columns=["Filename", "Language", "Text"])
    print("Parsing xml for", num_of_writings, "documents")
    start = time()
    df = DataFrame(Parallel(n_jobs=cpu_count(),
                            verbose=1)(delayed(parse_xml)(index) for index, row in df.iterrows()),
                   columns=["Filename", "Language", "Text"])
    df['Language'].replace({'cn': 'Chinese', 'de': 'German', 'fr': 'French', 'it': 'Italian', 'jp': 'Japanese',
                            'kr': 'Korean', 'mx': 'Spanish', 'sa': 'Arabic', 'tr': 'Turkish'}, inplace=True)

    print("Parsed", num_of_writings, "in", (time() - start)/60, "minutes")
    print("Writing data to tsv file")
    df.to_csv("efcamdat_tokenized.tsv", sep='\t', index=False)


if __name__ == '__main__':
    main()
