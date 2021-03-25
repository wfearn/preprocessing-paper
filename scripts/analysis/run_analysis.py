from preprocess import analyzer, argmanager
import sys

methods_index = list(sys.argv).index('-m')

# Checks to see if there are methods
if len(sys.argv) == (methods_index) + 1:
    del sys.argv[methods_index]

args = argmanager.parse_args()
analyzer.analyze(args.corpus, args.methods, args.model, args.train, args.seed)
