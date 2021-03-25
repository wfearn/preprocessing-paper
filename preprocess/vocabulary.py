import numpy as np
import json
import pickle
import os
from preprocess import argmanager as arg
from collections import namedtuple, Counter, defaultdict

vocab_dir = os.path.join(os.getenv('HOME'), '.preprocess/vocabulary')

def create_bpe_set_filename(corpus, methods):
    if type(corpus) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')
    if corpus == '':
        raise AttributeError('Empty string invalid for parameter \'corpus\'')

    base_filename = f'{arg.processed_corpus_name(corpus, methods)}_bpe_set.pickle'
    return os.path.join(vocab_dir, f'{corpus}/{base_filename}')

def create_vocabulary_filename(corpus, methods, seed):
    if type(corpus) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')
    if type(seed) is not int:
        raise TypeError('Invalid non-int type for parameter \'seed\'')
    if corpus == '':
        raise AttributeError('Empty string invalid for parameter \'corpus\'')

    base_filename = f'{arg.processed_corpus_name(corpus, methods)}'

    # Seed only affects hashing values
    # We expect this to be called from a parallelized program
    # So it should receive a seed parameter in most cases
    # and it will not always be necessary
    if not perform_hashing(methods):
        base_filename = f'{base_filename}_vocabulary.pickle'
    else:
        base_filename = f'{base_filename}_s{seed}_vocabulary.pickle'

    return os.path.join(vocab_dir, f'{corpus}/{base_filename}')

def perform_bpe(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')

    for method in methods:
        # We expect method name to come in 'h(num)' format
        if arg.is_bpe_method(method):
            return True
    return False

def perform_hashing(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')

    for method in methods:
        # We expect method name to come in 'h(num)' format
        if arg.is_hashing_method(method):
            return True
    return False

def bpe_value(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')

    for method in methods:
        if arg.is_bpe_method(method):
            return int(method[3:])

    raise AttributeError('Parameter \'methods\' had no valid bpe option')

def hashing_value(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')

    for method in methods:
        if arg.is_hashing_method(method):
            return int(method[1:])

    raise AttributeError('Parameter \'methods\' had no valid hashing option')

def retrieve_pickled_vocabulary(vocab_filename):
    if type(vocab_filename) is not str:
        raise TypeError('Invalid non-string type for parameter \'vocab_filename\'')
    if not os.path.exists(vocab_filename):
        raise AttributeError(f'Invalid filepath: {vocab_filename}')

    with open(vocab_filename, 'rb') as p:
        pv = pickle.load(p)

    return pv

def retrieve_vocabulary(corpus, methods, seed):
    if type(corpus) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus\'')
    if corpus == '':
        raise AttributeError('Invalid empty string for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')
    if type(seed) is not int:
        raise TypeError('Invalid non-int type for parameter \'seed\'')

    if not os.path.isdir(vocab_dir):
        os.mkdir(vocab_dir)
        for valid_corpus in arg.valid_corpora:
            os.mkdir(os.path.join(vocab_dir, valid_corpus))

    vf = create_vocabulary_filename(corpus, methods, seed)

    if os.path.exists(vf):
        return retrieve_pickled_vocabulary(vf)

    return generate_vocabulary(corpus, methods, vf)

def check_json(corpus_filepath):
    if type(corpus_filepath) is not str:
        raise TypeError('Invalid non-string value')
    if corpus_filepath == '':
        raise AttributeError('Invalid empty string for parameter corpus_filepath')

    return 'json' in corpus_filepath

def retrieve_text(s, is_json, tag='reviewText'):
    if type(s) is not str:
        raise TypeError('Invalid non-string value for parameter s')
    if type(is_json) is not bool:
        raise TypeError('Invalid non-bool value for parameter is_json')
    if type(tag) is not str:
        raise TypeError('Invalid non-string value for parameter tag')
    if s == '':
        raise AttributeError('Invalid empty string value for parameter s')
    if is_json and not tag:
        raise AttributeError('Invalid empty string for parameter \'tag\' when is_json is true')

    return json.loads(s)[tag] if is_json else s

# This isn't used here but it is associated with vocabulary-type things so its in
# this file
def generate_stopword_list():
    #stopword_filepath = os.path.join(os.getenv('HOME'), 'preprocess/utilities/english.txt')
    #assert os.path.exists(stopword_filepath)

    #with open(stopword_filepath, 'r') as f:
    #    words = { w.strip('\n') for w in f.readlines() }

    from nltk.corpus import stopwords

    stopword_list = stopwords.words('english')

    for word in stopwords.words('english'):
        if "'" in word:
            w = word.replace("'", '')
            stopword_list.append(w)

    return set(stopword_list)

def retrieve_bpe_set(corpus, methods):
    bpe_set_filename = create_bpe_set_filename(corpus, methods)

    if os.path.isfile(bpe_set_filename):
        with open(bpe_set_filename, 'rb') as f:
            bpe_set = pickle.load(f)
    else:
        corpus_filename = arg.retrieve_corpus_file(corpus, methods)
        is_json = check_json(corpus_filename)
        text_tag = arg.corpus_text_dict[corpus]

        bpe_set = create_bpe_set(corpus, methods, is_json, text_tag, bpe_value(methods))

    return bpe_set

# Implements word BPE as described by (Sennrich, et. al 2016)
# Note there are libraries for this already but I wanted
# something that used a dictionary of frequencies due to corpus size
# and I wanted control over whether there were word boundaries or not.
def create_bpe_set(corpus, methods, is_json, tag, vocab_size, reduce_vocab=True):
    if type(corpus) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus\'')
    if corpus == '':
        raise AttributeError('Invalid empty string for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')
    if type(is_json) is not bool:
        raise TypeError('Invalid non-bool value for parameter is_json')
    if type(tag) is not str:
        raise TypeError('Invalid non-string value for parameter tag')
    if type(vocab_size) is not int:
        raise TypeError('Invalid non-int value for parameter vocab_size')

    partial_vocab = 0.95
    full_vocab = 1.0

    import gzip
    import re

    vocab = Counter()
    import gzip

    with gzip.open(arg.retrieve_corpus_file(corpus, methods), 'rb') as f:
        for line in f:
            words = retrieve_text(line.decode('utf-8'), is_json, tag=tag).split()
            for word in words:
                vocab.update([' '.join([c for c in word])])

    percentage_break = partial_vocab if reduce_vocab else full_vocab

    reduced_vocab = dict()
    total_tokens = np.sum(list(vocab.values()))
    tokens_so_far = int(0)

    for (term, frequency) in vocab.most_common():
        tokens_so_far += frequency
        percent = tokens_so_far / total_tokens
        reduced_vocab[term] = frequency

        if percent > percentage_break: break

    def get_stats(vocab):
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        return pairs

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out


    char_set = set(''.join(' '.join(reduced_vocab.keys()).split()))
    num_merges = vocab_size - len(char_set)

    for _ in range(num_merges):
        pairs = get_stats(reduced_vocab)
        best = max(pairs, key=pairs.get)
        reduced_vocab = merge_vocab(best, reduced_vocab)

    bpe_set = set(' '.join(reduced_vocab.keys()).split())

    bpe_set_filename = create_bpe_set_filename(corpus, methods)

    with open(bpe_set_filename, 'wb') as f:
        pickle.dump(bpe_set, f, protocol=4)

    return bpe_set


# Will I be calling this function from any other function other than the retrieve vocabulary one?
# Might want to put in a case where it returns immediately if the vocabulary file already exists
# TODO: Remove vocab_filename parameter (not useful)
def generate_vocabulary(corpus, methods, vocab_filename):
    if type(corpus) is not str:
        raise TypeError('Invalid non-string type for parameter \'corpus\'')
    if corpus == '':
        raise AttributeError('Invalid empty string for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')
    if type(vocab_filename) is not str:
        raise TypeError('Invalid non-string type for parameter \'vocab_filename\'')
    if vocab_filename == '':
        raise AttributeError('Invalid empty string for parameter \'vocab_filename\'')

    corpus_filename = arg.retrieve_corpus_file(corpus, methods)

    if not os.path.isfile(corpus_filename):
        raise ValueError(f'{corpus_filename} does not exist, must generate before building vocabulary')

    is_json = check_json(corpus_filename)
    text_tag = arg.corpus_text_dict[corpus]

    if perform_hashing(methods):
        v = HashVocabInfo(hashing_value(methods))
#    elif perform_bpe(methods):
#        bpe_set = create_bpe_set(corpus, methods, is_json, text_tag, bpe_value(methods))
#        v = BPEVocabInfo(bpe_set)
    else:
        v = VocabInfo()

    #v = VocabInfo() if not perform_hashing(methods) else HashVocabInfo(hashing_value(methods))

    import gzip
    with gzip.open(arg.retrieve_corpus_file(corpus, methods), 'rb') as f:
        for line in f:
            words = retrieve_text(line.decode('utf-8'), is_json, tag=text_tag).split()
            for word in set(words):
                v.increment_doc_frequency(word)

            for word in words:
                v.increment_term_frequency(word)

    with open(vocab_filename, 'wb') as f:
        pickle.dump(v, f, protocol=4)

    return v

# Can I put these inside of the class definition or anything?
NUM_INDEX = 0
TF_INDEX = 1
DF_INDEX = 2

# Keeps track of vocabulary but also global term/document frequency
class VocabInfo:
    def __init__(self):
        self.vocabulary = dict()

    def _place_in_dict(self, token):
        if type(token) is not str:
            raise TypeError('Non-string value not a valid key')

        if token not in self.vocabulary.keys():
            self.vocabulary[token] = [np.uint32(len(self.vocabulary)), np.uint32(0), np.uint32(0)]

    @property
    def dictionary(self):
        return self.vocabulary

    @dictionary.setter
    def dictionary(self, d):
        self.vocabulary = d

    def __getitem__(self, token):
        self._place_in_dict(token)
        return self.vocabulary[token][NUM_INDEX]

    def increment_term_frequency(self, token):
        self._place_in_dict(token)
        self.vocabulary[token][TF_INDEX] += 1

    def increment_doc_frequency(self, token):
        self._place_in_dict(token)
        self.vocabulary[token][DF_INDEX] += 1

    def term_frequency(self, token):
        self._place_in_dict(token)
        return self.vocabulary[token][TF_INDEX]

    def doc_frequency(self, token):
        self._place_in_dict(token)
        return self.vocabulary[token][DF_INDEX]

    def __len__(self):
        return len(self.vocabulary)

class HashVocabInfo(VocabInfo):
    def __init__(self, size):
        super(HashVocabInfo, self).__init__()
        self.size = size
        self.types = set()

    def _place_in_dict(self, token):
        if type(token) is not str:
            raise TypeError('Non-string value not a valid key')

        if token not in self.vocabulary.keys():
            num = hash(token) % self.size
            self.types.update([num])
            self.vocabulary[token] = [np.uint32(num), np.uint32(0), np.uint32(0)]

    def __len__(self):
        return len(self.types)

class BPEVocabInfo(VocabInfo):
    def __init__(self, vocab_set):
        super(BPEVocabInfo, self).__init__()
        self.vocab_set = set()

    def _place_in_dict(self, token):
        token_segments = self.segment_token(token)

        for segment in token_segments:
            super()._place_in_dict(segment)

    def segment_token(self, token):
        segments = list()

        while len(token) > 1:
            for i in reversed(range(1, len(token) + 1)):
                subword = token[:i]
                if subword in self.vocab_set:
                    segments.append(subword)

                token = token[i:]

        if len(token):
            segments.append(token)

        return segments

    def increment_term_frequency(self, token):
        token_segments = self.segment_token(token)

        for segment in token_segments:
            self._place_in_dict(segment)
            self.vocabulary[segment][TF_INDEX] += 1

    def increment_doc_frequency(self, token):
        token_segments = self.segment_token(token)

        for segment in token_segments:
            self._place_in_dict(segment)
            self.vocabulary[segment][DF_INDEX] += 1

    # The function calls for this should assume pre-segmented tokens
    def term_frequency(self, token):
        assert token in self.vocab_set
        return self.vocabulary[token][TF_INDEX]

    def doc_frequency(self, token):
        assert token in self.vocab_set
        return self.vocabulary[token][DF_INDEX]

    def __len__(self):
        return len(self.vocab_set)
