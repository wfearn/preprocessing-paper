import ankura
import json
import string
import re

corpus_environment_name = 'PREPROCESS_CORPUS'
methods_environment_name = 'PREPROCESS_METHODS'

def load_stopword_list():
    raise NotImplementedError

def string_check(fn):
    def wrapper(s):
        if s is None:
            raise AttributeError('NoneType is not valid')
        elif type(s) != str:
            raise AttributeError(f'Type: {type(s)} not valid')

        return fn(s)
    return wrapper

# Never use kwargs just need it so interface
# Between plaintext and json processor is the same.
def plaintext_processor(processor, **kwargs):

    def text_processor(s):
        return processor(s)
    return text_processor

def json_processor(processor, tag='reviewText'):
    import json

    p = plaintext_processor(processor)
    def text_processor(s):
        obj = json.loads(s)
        result = p(obj[tag])
        obj[tag] = result
        return json.dumps(obj)
    return text_processor

def lowercase_preprocessor():

    @string_check
    def lower(s):
        return s.lower()
    return lower

def bpe_encode(word, bpe_set):
    splits = list()

    while len(word) > 0:

        for i in range(len(word)):
            partial_word = word[:-i] if i else word[i:]
            if len(partial_word) < 2 or partial_word in bpe_set:
                splits.append(partial_word)

                # If the whole word is in bpe_set i will be 0
                word = word[-i:] if i else word[:i]
                break

    return splits

def bpe_preprocessor():
    import os
    from preprocess import vocabulary
    from preprocess import argmanager

    # THE HACKIEST
    corpus = os.getenv(corpus_environment_name)
    methods = os.getenv(methods_environment_name)

    # getenv returns none if there is no value assigned
    assert corpus
    assert methods

    bpe_set = vocabulary.retrieve_bpe_set(corpus, [str(m) for m in methods.split(',')])

    @string_check
    def bpe_encoder(s):
        new_words = list()
        new_words.extend(bpe_encode(s, bpe_set))
        return ' '.join(new_words)

    return bpe_encoder

def segment_preprocessor():
    from wordsegment import load, segment
    load()

    @string_check
    def segmenter(s):
        new_words = list()
        for word in s.split():
            try:
                new_words.extend(segment(word))
            except RecursionError:
                new_words.extend([word])

        return ' '.join(new_words)
    return segmenter

def spelling_preprocessor():
    import os
    from symspellpy.symspellpy import SymSpell, Verbosity

    max_edit_distance_dictionary = 2
    prefix_length = 7

    sc = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = os.path.join(os.getenv('HOME'), 'symspellpy/symspellpy/frequency_dictionary_en_82_765.txt')
    term_index = 0
    count_index = 1

    if not sc.load_dictionary(dictionary_path, term_index, count_index):
        raise ImportError('Unable to load spelling dictionary')

    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST

    @string_check
    def checker(s):
        words = s.split()
        corrected_words = list()

        for word in words:
            correction = sc.lookup(word, suggestion_verbosity, max_edit_distance_lookup)
            if correction:
                corrected_words.append(correction[0].term)
            else:
                corrected_words.append(word)
        return ' '.join(corrected_words)
    return checker

def dash_underscore_preprocessor():
    udrep = re.compile('[_|-]+')

    @string_check
    def replacer(s):
        return ' '.join([st.strip() for st in udrep.sub(' ', s).split() if st != ''])
    return replacer

def lemmatizing_preprocessor():
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()

    @string_check
    def lemmatizer(s):
        return ' '.join([lem.lemmatize(st) for st in s.split()])
    return lemmatizer

def punctuation_preprocessor():

    @string_check
    def replacer(s):
        delp_table = str.maketrans('', '', string.punctuation)
        return s.translate(delp_table)
    return replacer

def stemming_preprocessor():
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    @string_check
    def stemmer(s):
        return ' '.join([ps.stem(st) for st in s.split()])
    return stemmer

def number_preprocessor():

    @string_check
    def remover(s):
        return re.sub('[\d]+', '', s)
    return remover

def base_preprocessor():
    #Assume whitespace is boundaries between words
    @string_check
    def base(s):
        return ' '.join([st.strip() for st in s.split() if st != ''])
    return base

# Have this in its own function for testing
def sort_methods(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list type for parameter \'methods\'')

    from collections import defaultdict

    # Order is important because we don't want to split and then correct
    # Or do segmentation after punctuation removal
    # Or do stemming and then correction
    # defaultdict is so low-priority methods such as r(num) default to 0
    preprocess_order_dict = defaultdict(int)
    preprocess_order_dict['np'] = 5
    preprocess_order_dict['nr'] = 5
    preprocess_order_dict['ud'] = 5
    preprocess_order_dict['ws'] = 4
    preprocess_order_dict['sc'] = 3
    preprocess_order_dict['st'] = 2
    preprocess_order_dict['lm'] = 2
    preprocess_order_dict['lc'] = 1

    return sorted(methods, key=lambda method: preprocess_order_dict[method], reverse=True)

def create_preprocessor(methods):

    preprocess_dict = {
                          'lc' : lowercase_preprocessor(),
                          'ws' : segment_preprocessor(),
                          'sc' : spelling_preprocessor(),
                          'ud' : dash_underscore_preprocessor(),
                          'st' : stemming_preprocessor(),
                          'lm' : lemmatizing_preprocessor(),
                          'np' : punctuation_preprocessor(),
                          'nr' : number_preprocessor(),
                      }

    b = base_preprocessor()

    methods = sort_methods(methods)
    from preprocess import argmanager as arg
    def preprocessor(s):

        s = b(s)
        for method in methods:
            #Skip stopword removal, hashing, rare word filtering, and test preprocessing
            if method == 'sp' or method == 'tt' or arg.is_rare_filter_method(method) or arg.is_hashing_method(method): continue

            if arg.is_bpe_method(method):
                bpep = bpe_preprocessor()
                s = bpep(s)

            s = preprocess_dict[method](s)

        return s
    return preprocessor
