import argparse
import os

corpus_tag_dict = {
                    'reddit'     : 'score',
                    'testreddit' : 'score',
                    'amazon'     : 'overall',
                    'testamazon' : 'overall',
                    'ap_news'    : 'year',
                    'apnews'     : '',
                    'testapnews' : '',
                    'twitter'    : '',
                    'testtwitter': '',
                    'notacorpus' : '',
                  }

corpus_text_dict = {
                         'reddit' : 'body',
                     'testreddit' : 'body',
                 'testredditjson' : 'body',
                         'amazon' : 'reviewText',
                     'testamazon' : 'reviewText',
                     'testjson'   : 'reviewText',
                      'ap_news'   : 'text',
                     'apnews'     : '',
                     'testapnews' : '',
                     'twitter'    : '',
                     'testtwitter': '',
                     'notacorpus' : '',
                   }

base_dir = os.path.join(os.getenv('HOME'), '.preprocess')
valid_corpora = ['notacorpus', 'reddit', 'apnews', 'amazon', 'testreddit', 'testapnews', 'testamazon', 'twitter', 'testtwitter', 'ap_news']
valid_models = ['svm', 'knn', 'naive', 'ankura', 'vowpal']

def parse_args():
    parser = argparse.ArgumentParser()

    # comma-separated list to enumerate pipeline options
    parser.add_argument('-m', '--methods', type=lambda s: [str(o) for o in s.split(',')], default=list())
    parser.add_argument('-c', '--corpus', type=str, choices=valid_corpora)
    parser.add_argument('-r', '--run', help='The run number', type=int, default=None)
    parser.add_argument('-o', '--offset', help='The offset from the starting line', type=int, default=None)
    parser.add_argument('-s', '--seed', help='The seed to run the algorithm with', type=int, default=0)
    parser.add_argument('-tr', '--train', help='The size of the training set', type=int, default=10000)
    parser.add_argument('-mo', '--model', help='The machine learning algorithm to use', type=str, choices=valid_models)
    parser.add_argument('-si', '--corpus_size', help='The number of documents in the corpus', type=int, default=None)
    return parser.parse_args()

def is_bpe_method(method):
    return 'bpe' in method and method[3:].isdigit()

def is_hashing_method(method):
    return 'h' in method and method[1:].isdigit()

def is_rare_filter_method(method):
    return 'r' in method and method[1:].isdigit()

def processed_corpus_name(corpus, methods):

    if 'sc' in methods: corpus = f'{corpus}_spell'
    if 'ws' in methods: corpus = f'{corpus}_seg'
    if 'lc' in methods and not ('sc' in methods or 'ws' in methods): corpus = f'{corpus}_lower'
    if 'np' in methods and not 'ws' in methods: corpus = f'{corpus}_nopunct'
    if 'ud' in methods: corpus = f'{corpus}_udrep'
    if 'nr' in methods: corpus = f'{corpus}_nrem'
    if 'sp' in methods: corpus = f'{corpus}_stop'
    if 'st' in methods: corpus = f'{corpus}_stem'
    if 'lm' in methods: corpus = f'{corpus}_lemma'
    if 'tt' in methods: corpus = f'{corpus}_test'

    # hashing
    for method in methods:
        if is_hashing_method(method):
            corpus = f'{corpus}_{method}'

    # rare word filtering
    for method in methods:
        if is_rare_filter_method(method):
            corpus = f'{corpus}_{method}'

    # byte pair encoding
    for method in methods:
        if is_bpe_method(method):
            corpus = f'{corpus}_{method}'

    return corpus

def results_dir(corpus):
    return f'{base_dir}/results/{corpus}'

def corpus_dir(corpus):
    return f'{base_dir}/corpora/{corpus}'

def retrieve_corpus_file(corpus, methods, run_number=None):

    ext_dict = {
                        'amazon' : 'json',
                    'testamazon' : 'json',
                        'reddit' : 'json',
                    'testreddit' : 'json',
                       'ap_news' : 'json',
                        'apnews' : 'txt',
                    'testapnews' : 'txt',
                       'twitter' : 'txt',
                   'testtwitter' : 'txt',
                    'notacorpus' : 'txt',
                      'testjson' : 'json',
                'testredditjson' : 'json',
               }

    ext = ext_dict[corpus]
    name = processed_corpus_name(corpus, methods)

    # Sometimes we're pulling corpora that exists, and sometimes we're creating it.
    name = f'{name}_{run_number}' if run_number is not None else name

    return f'{corpus_dir(corpus)}/{name}.{ext}.gz'
