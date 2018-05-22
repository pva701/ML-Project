import re

import numpy as np
import json
import pickle


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def dump_json_word_vecs_np(fname, word_vecs):
    word2vec_list = {}
    for k, v in word_vecs.items():
        word2vec_list[k] = v.tolist()

    with open(fname, 'w') as f:
        json.dump(word2vec_list, f)


def load_json_word_vecs_np(fname):
    with open(fname, 'r') as f:
        word2vec_list = json.load(f)
        word2vec_np = {}
        for k, v in word2vec_list.items():
            word2vec_np[k] = np.array(v, dtype=np.float32)
        return word2vec_np


def dump_pickle_word_vecs_np(fname, word_vecs):
    with open(fname, 'wb') as f:
        pickle.dump(word_vecs, f)


def load_pickle_word_vecs_np(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def add_unknown_words(word_vecs, vocab_dict, bound=0.25, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab_dict:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-bound, bound, k)
