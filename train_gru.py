#! /usr/bin/env python

import os
import time
from datetime import datetime

import utils
from gru import GRU
from utils import *
from flags import FLAGS

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE"
                                     "", "8000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
print(INPUT_DATA_FILE)
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE, max_sents=1000000)

if not FLAGS.print_sentences:
    # Build model
    model = GRU(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

    # Print SGD step time
    def sgd_callback(model, num_examples_seen):
        dt = datetime.now().isoformat()
        loss = model.calculate_loss(x_train[:10000], y_train[:10000])
        print("\n%s (%d)" % (dt, num_examples_seen))
        print("--------------------------------------------------")
        print("Loss: %f" % loss)
        generate_sentences_from_scratch(model, 10, index_to_word, word_to_index)
        save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
        print("\n")
        sys.stdout.flush()

    for epoch in range(NEPOCH):
      train_with_sgd(model, x_train, y_train,
                     learning_rate=LEARNING_RATE,
                     nepoch=1,
                     decay=0.9,
                     callback_every=PRINT_EVERY,
                     callback=sgd_callback)
else:
    model = utils.load_model_parameters_theano("data/pretrained-theano.npz")
    generate_sentences_from_scratch(model, 100, index_to_word, word_to_index)

    print(" == Prefixed sentences == ")
    generate_sentences_by_prefixes(
        model
        , [
            ["i", "am"]
        ,   ["robots", "stop"]
        ,   ["it", "depends"]
        ,   ["i", "am", "going"]
        ,   ["you", "are", "ridiculous"]
        ]
        , index_to_word, word_to_index)