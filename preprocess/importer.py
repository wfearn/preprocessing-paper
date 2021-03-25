from collections import namedtuple
import pickle
import json
import numpy as np
import gzip
from preprocess import vocabulary as v
from preprocess import argmanager as arg
from preprocess import sample as s
import os

Token = namedtuple('Token', 'token')
Document = namedtuple('Document', 'tokens metadata')
Corpus = namedtuple('Corpus', 'documents vocabulary metadata')

import_corpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/imported_corpora')

DEFAULT_TEST_SIZE = 1000

class FilterDict(dict):
    def __missing__(self, val):
        return lambda x : x

filter_dict = FilterDict()
filter_dict['reddit'] = lambda x : 1 if x > 1 else 0
filter_dict['amazon'] = lambda x : 1 if x > 4 else 0
filter_dict['ap_news'] = lambda x : 1 if int(x) <= 96 else 0

def create_imported_corpus_filename(corpus, methods, train_size, seed):
    if type(corpus) is not str:
        raise TypeError('Non-string value invalid for parameter \'corpus\'')
    if not corpus:
        raise AttributeError('Parameter \'corpus\' must be non-empty')
    if type(methods) is not list:
        raise TypeError('Non-list value invalid for parameter \'methods\'')
    if type(train_size) is not int:
        raise TypeError('Non-int value invalid for parameter \'train_size\'')
    if type(seed) is not int:
        raise TypeError('Non-int value invalid for parameter \'seed\'')
    if train_size <= 0:
        raise AttributeError('Non-positive values invalid for parameter \'train_size\'')
    if seed < 0:
        raise AttributeError('Negative values invalid for parameter \'seed\'')

    from preprocess import argmanager as arg

    base_filename = f'{arg.processed_corpus_name(corpus, methods)}_s{seed}_corpus{train_size}.pickle'
    return os.path.join(import_corpus_dir, f'{corpus}/{base_filename}')

def is_corpus_testable(corpus):
    if type(corpus) is not str:
        raise TypeError('Non-string value invalid for parameter \'corpus\'')
    if not corpus:
        raise AttributeError('Parameter \'corpus\' must be non-empty')

    testable_corpora = { 'reddit', 'amazon', 'testamazon', 'testreddit', 'ap_news'}
    return corpus in testable_corpora

def get_target(document, tag='overall'):
    if type(document) is not str:
        raise TypeError('Invalid non-string type for parameter \'document\'')
    if not document:
        raise AttributeError('Invalid empty string for parameter \'document\'')
    if type(tag) is not str:
        raise TypeError('Invalid non-string type for parameter \'tag\'')
    if not tag:
        raise AttributeError('Invalid empty string for parameter \'tag\'')

    return int(json.loads(document)[tag])

def retrieve_preprocessed_corpus(corpus, methods):
    if type(corpus) is not str:
        raise TypeError('Non-string value invalid for parameter \'corpus\'')
    if not corpus:
        raise AttributeError('Parameter \'corpus\' must be non-empty')
    if type(methods) is not list:
        raise TypeError('Non-list value invalid for parameter \'methods\'')
    documents = list()

    corpus_file = arg.retrieve_corpus_file(corpus, methods)
    with gzip.open(corpus_file, 'rb') as f:
        for line in f:
            documents.append(line.decode('utf-8').strip('\n'))

    return documents

def extract_documents(subset, testable, vocabulary, reject, corpus_name, train_vocabulary=None):
    if type(subset) is not list:
        raise TypeError('Non-list type invalid for parameter \'subset\'')
    if type(testable) is not bool:
        raise TypeError('Non-bool type invalid for parameter \'testable\'')
    if type(vocabulary) is not v.VocabInfo and type(vocabulary) is not v.HashVocabInfo:
        raise TypeError('Non-VocabInfo type invalid for parameter \'vocabulary\'')
    if type(corpus_name) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus_name\'')
    if not subset:
        raise AttributeError('Empty list invalid for parameter \'subset\'')
    if not corpus_name:
        raise AttributeError('Empty string invalid for parameter \'corpus_name\'')

    documents = list()
    total_tokens = int(0)
    small_vocab = dict() if train_vocabulary is None else train_vocabulary
    tag_dict = arg.corpus_tag_dict
    text_dict = arg.corpus_text_dict
    target_filter = filter_dict[corpus_name]

    for doc in subset:
        # No empty documents
        if not doc: continue

        tokens = list()
        target = target_filter(get_target(doc, tag=tag_dict[corpus_name])) if testable else np.random.randint(2)

        # For the moment being testable is synonymous with being a json document
        for word in v.retrieve_text(doc, testable, tag=text_dict[corpus_name]).split():
            if reject(word): continue
            w_t = vocabulary[word]

            # We do this in the train set to avoid tokens larger than the vocab size
            if w_t not in small_vocab.keys():
                # If train_vocabulary is not None this should never be hit
                assert train_vocabulary is None
                small_vocab[w_t] = len(small_vocab)

            total_tokens += 1
            token = np.uint32(small_vocab[w_t])
            tokens.append(Token(token))

        # Filter out empty documents
        if not tokens: continue

        # Putting in metadata dictionary to use assignment strategies
        # with ankura should we so choose
        documents.append(Document(tokens, dict({'target' : target})))

    # We return small vocab for use by test set
    return documents, total_tokens, small_vocab

def retrieve_imported_corpus(corpus, methods, train_size, seed):
    if type(corpus) is not str:
        raise TypeError('Non-string value invalid for parameter \'corpus\'')
    if not corpus:
        raise AttributeError('Parameter \'corpus\' must be non-empty')
    if type(methods) is not list:
        raise TypeError('Non-list value invalid for parameter \'methods\'')
    if type(train_size) is not int:
        raise TypeError('Non-int value invalid for parameter \'train_size\'')
    if type(seed) is not int:
        raise TypeError('Non-int value invalid for parameter \'seed\'')
    if train_size <= 0:
        raise AttributeError('Non-positive values invalid for parameter \'train_size\'')
    if seed < 0:
        raise AttributeError('Negative values invalid for parameter \'seed\'')

    corpus_filename = create_imported_corpus_filename(corpus, methods, train_size, seed)
    with open(corpus_filename, 'rb') as f:
        imported_corpus = pickle.load(f)

    return imported_corpus

def create_imported_corpus_dirs():
    from preprocess import argmanager as arg

    if not os.path.isdir(import_corpus_dir):
        os.mkdir(import_corpus_dir)

    for c in arg.valid_corpora:
        dir = os.path.join(import_corpus_dir, c)
        if not os.path.isdir(dir):
            os.mkdir(dir)

def import_corpus(corpus, methods, train_size, seed, test_size=DEFAULT_TEST_SIZE):

    np.random.seed(seed)

    imported_corpus_filename = create_imported_corpus_filename(corpus, methods, train_size, seed)
    if os.path.isfile(imported_corpus_filename):
        train, test = retrieve_imported_corpus(corpus, methods, train_size, seed)
        return train, test

    if not os.path.isdir(import_corpus_dir):
        create_imported_corpus_dirs()

    large_vocab = v.retrieve_vocabulary(corpus, methods, seed)
    all_documents = retrieve_preprocessed_corpus(corpus, methods)
    np.random.shuffle(all_documents)

    subset = all_documents[:train_size + test_size]
    testable = is_corpus_testable(corpus)
    rare_filter = s.get_rare_filter(methods)
    stopword_list = v.generate_stopword_list() if s.perform_stopword_filtering(methods) else set()
    small_v = set()
    metadata = dict()

    train_docs, total_tokens, small_v  = extract_documents(subset[:train_size],
                                            testable,
                                            large_vocab,
                                            lambda x : large_vocab.doc_frequency(x) < rare_filter or x in stopword_list,
                                            corpus)

    metadata['total_tokens'] = total_tokens

    test_docs, _, _ = extract_documents(subset[train_size : test_size + train_size],
                         testable,
                         large_vocab,
                         lambda x : large_vocab[x] not in small_v,
                         corpus,
                         train_vocabulary=small_v)

    # Corpora are structured similarly to ankura corpora to allow interoperability between
    # Ankura and sklearn
    train = Corpus(train_docs, set(small_v.keys()), metadata)
    test = Corpus(test_docs, {}, {})

    with open(imported_corpus_filename, 'wb') as f:
        pickle.dump((train, test), f)

    return train, test
