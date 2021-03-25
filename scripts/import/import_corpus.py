#!/usr/bin/python

from preprocess import argmanager, importer
import sys

methods_index = list(sys.argv).index('-m')

# Checks to see if there are methods
if len(sys.argv) == (methods_index) + 1:
    del sys.argv[methods_index]

args = argmanager.parse_args()
importer.import_corpus(args.corpus, args.methods, args.train, args.seed)
