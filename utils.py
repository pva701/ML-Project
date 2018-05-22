#! /usr/bin/env python

import csv
import itertools
import operator
import sys

import nltk
import numpy as np

from gru import GRU

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"


def load_data(filename, vocabulary_size, min_sent_characters=1, max_sents=20000):
    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        sentences = sentences[0:min(len(sentences), max_sents)]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size - 2]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train, word_to_index, index_to_word


def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
                   callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)
    return model


def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
             E=model.E.get_value(),
             U=model.U.get_value(),
             W=model.W.get_value(),
             V=model.V.get_value(),
             b=model.b.get_value(),
             c=model.c.get_value())
    print "Saved model parameters to %s." % outfile


def load_model_parameters_theano(path, modelClass=GRU):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def fill(seq, l):
    r = []
    for x in seq:
        while len(x) < l:
            x.append(0)
        r.append(x)
    return np.asarray(r)


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()


def generate_sentence_from_scratch(model, index_to_word, word_to_index, min_length=5):
    return generate_sentence_by_prefix(model, [], index_to_word, word_to_index, min_length)


def generate_sentence_by_prefix(model, prefix, index_to_word, word_to_index, min_length=5):
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]] + [word_to_index[w] for w in prefix]
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence


def generate_sentences_from_scratch(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence_from_scratch(model, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)


def generate_sentences_by_prefixes(model, prefixes, index_to_word, word_to_index):
    for prefix in prefixes:
        sent = None
        while not sent:
            sent = generate_sentence_by_prefix(model, prefix, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)
