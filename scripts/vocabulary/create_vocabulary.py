#!/usr/bin/python

from preprocess import argmanager, vocabulary
import sys

methods_index = list(sys.argv).index('-m')

# Checks to see if there are methods
if len(sys.argv) == (methods_index) + 1:
    del sys.argv[methods_index]

args = argmanager.parse_args()

sys.stdout.flush()

if not vocabulary.perform_hashing(args.methods) and args.seed > 0:
    print('No need for multiple seeds!')
    sys.stdout.flush()
    sys.exit()

vocabulary.retrieve_vocabulary(args.corpus, args.methods, args.seed)
