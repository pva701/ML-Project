__author__ = 'pva701'

import tensorflow as tf
import argparse

import sys

args = argparse.ArgumentParser()


# Parameters
# ==================================================

# Data loading params

args.add_argument("--dev_sample_percentage", type=float, default=.0)

# Model Hyperparameters
args.add_argument("--embedding_dim", type=int, default=128)
args.add_argument("--l2_reg_lambda", type=float, default=0.0)

# Training parameters
args.add_argument("--batch_size", type=int, default=10)
args.add_argument("--num_epochs", type=int, default=200)
args.add_argument("--evaluate_every", type=int, default=100)
args.add_argument("--checkpoint_every", type=int, default=100)
args.add_argument("--num_checkpoints", type=int, default=5)

# Misc Parameters
args.add_argument("--allow_soft_placement", type=bool, default=True)
args.add_argument("--log_device_placement", type=bool, default=False)
args.add_argument("--print-sentences", type=bool, default=False)

FLAGS, unparsed = args.parse_known_args()