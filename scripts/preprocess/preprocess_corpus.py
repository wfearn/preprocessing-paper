#!/usr/bin/python

from preprocess import argmanager, preprocessor
import sys

args = argmanager.parse_args()
preprocessor.preprocess_corpus(args.corpus, args.methods, run_number=args.run, offset=args.offset)
