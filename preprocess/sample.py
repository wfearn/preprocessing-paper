import numpy as np
import os
import pickle
from preprocess import vocabulary

TYPE_INDEX = 0
TOKEN_INDEX = 1
# Type counts and token counts
TYPES_OF_MEASUREMENT = 2
DEFAULT_STEP_SIZE = 1000

# Works on assumption that there will be no seeds higher than
# 10, if this changes this function needs to be changed.
def get_relevant_sample_files(corpus, methods, max_seeds=20):
    if type(corpus) is not str:
        raise TypeError('Non-string value not accepted for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Non-list value not accepted for parameter \'methods\'')
    if not corpus:
        raise AttributeError('Empty string invalid value for parameter \'corpus\'')

    files = set()

    for i in range(max_seeds):
        f = vocabulary.create_vocabulary_filename(corpus, methods, i)
        if os.path.isfile(f):
            files.update([f])

    return files

#TODO: Add provisions for sample size
def create_sample_filename(corpus, methods):
    if type(corpus) is not str:
        raise TypeError('Non-string value not accepted for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Non-list value not accepted for parameter \'methods\'')
    if not corpus:
        raise AttributeError('Empty string invalid value for parameter \'corpus\'')

    import preprocess
    path_elements = preprocess.argmanager.retrieve_corpus_file(corpus, methods).split('/')

    if 'corpora' not in path_elements:
        raise AttributeError(f'Corpus file path must be valid corpus path: {corpus_filepath}')

    path_elements[path_elements.index('corpora')] = 'samples'
    new_path = '/'.join(path_elements)

    from preprocess import combiner
    base, _ = combiner.get_base_and_ext(new_path)

    return f'{base}_sample.pickle'

def perform_stopword_filtering(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list value for parameter \'methods\'')

    return 'sp' in methods

def get_rare_filter(methods):
    if type(methods) is not list:
        raise TypeError('Invalid non-list value for parameter \'methods\'')

    import re
    rare_filter = re.compile(r'r(\d+)')

    filter_value = 1
    for method in methods:
        filter_values = rare_filter.findall(method)
        if filter_values:
            filter_value = int(filter_values[0])

    if filter_value <= 0:
        raise AttributeError('Rare filter value must be positive')

    return filter_value

def create_categorical(vocab):
    if type(vocab) is not vocabulary.VocabInfo and type(vocab) is not vocabulary.HashVocabInfo:
        raise TypeError('Non-VocabInfo type not accepted for parameter \'vocab\'')

    sample_values = list(vocab.dictionary.keys())
    p_values = np.asarray([vocab.term_frequency(word) for word in sample_values])
    p_values = p_values / np.sum(p_values)

    return sample_values, p_values

def sample_from_vocabulary(vocab, num_samples, sample_size, stop_filter=False, rare_filter=1, step_size=DEFAULT_STEP_SIZE):
    if type(vocab) is not vocabulary.VocabInfo and type(vocab) is not vocabulary.HashVocabInfo:
        raise TypeError('Non-VocabInfo value not accepted for parameter \'vocab\'')
    if type(num_samples) is not int:
        raise TypeError('Non-int value not accepted for parameter \'samples\'')
    if type(sample_size) is not int:
        raise TypeError('Non-int value not accepted for parameter \'sample_size\'')
    if type(stop_filter) is not bool:
        raise TypeError('Non-bool value not accepted for parameter \'stop_filter\'')
    if type(rare_filter) is not int:
        raise TypeError('Non-int value not accepted for parameter \'rare_filter\'')
    if rare_filter < 1:
        raise AttributeError('Parameter \'rare_filter\' must be positive')
    if num_samples < 0:
        raise AttributeError('Negative value not accepted for parameter \'samples\'')
    if sample_size < 0:
        raise AttributeError('Negative value not accepted for parameter \'sample_size\'')

    num_index = vocabulary.NUM_INDEX

    # For rare word filtering
    df_index = vocabulary.DF_INDEX
    stopword_list = vocabulary.generate_stopword_list() if stop_filter else set()

    sample_values, p_values = create_categorical(vocab)

    # Need an extra 1 because of last sample
    num_measurements = (sample_size // step_size) + 1
    results = np.zeros((num_samples, num_measurements, TYPES_OF_MEASUREMENT), dtype=np.uint32)

    for i in range(num_samples):
        types = set()

        # sample size + 1 to not miss last sample
        samples = np.random.choice(list(range(len(sample_values))), size=(sample_size + 1), p=p_values, replace=True)

        for j, s_i in enumerate(samples):

            word = sample_values[s_i]
            if word not in stopword_list and vocab.doc_frequency(word) >= rare_filter:
                types.update([vocab[word]])

            if not (j % step_size):
                measurement_index = j // step_size
                results[i][measurement_index][TYPE_INDEX] = len(types)
                # j keeps track nicely of how many tokens we've looked at
                results[i][measurement_index][TOKEN_INDEX] = j + 1

    return np.mean(results, axis=0)

def retrieve_sample_pickle(sample_filename):
    if type(sample_filename) is not str:
        raise TypeError('Non-string valid not accepted for parameter \'sample_filename\'')
    if not sample_filename:
        raise AttributeError('Parameter \'sample_filename\' must be non-empty')
    if not os.path.isfile(sample_filename):
        raise AttributeError(f'Nonexistant file {sample_filename}')

    import pickle
    with open(sample_filename, 'rb') as f:
        p = pickle.load(f)

    return p

def create_sample_dirs(sample_dir):
    if type(sample_dir) is not str:
        raise TypeError('Non-string valid not accepted for parameter \'sample_dir\'')
    if not sample_dir:
        raise AttributeError('Parameter \'sample_dir\' must be non-empty')
    os.mkdir(sample_dir)

    from preprocess import argmanager
    for valid_corpora in argmanager.valid_corpora:
        os.mkdir(os.path.join(sample_dir, f'{valid_corpora}'))

def sample(corpus, methods, samples=5, sample_size=500000000):
    if type(corpus) is not str:
        raise TypeError('Non-string value not accepted for parameter \'corpus\'')
    if type(methods) is not list:
        raise TypeError('Non-list value not accepted for parameter \'methods\'')
    if not corpus:
        raise AttributeError('Empty string invalid value for parameter \'corpus\'')

    sample_filename = create_sample_filename(corpus, methods)
    if os.path.isfile(sample_filename):
        return retrieve_sample_pickle(sample_filename)

    v = vocabulary.retrieve_pickled_vocabulary

    sample_dir = os.path.join(os.getenv('HOME'), '.preprocess/samples')
    if not os.path.isdir(sample_dir):
        create_sample_dirs(sample_dir)

    files = get_relevant_sample_files(corpus, methods)

    # Need an extra 1 because of last sample in sample_from_vocabulary
    num_measurements = (sample_size // DEFAULT_STEP_SIZE) + 1
    vocabulary_results = np.zeros((len(files), num_measurements, TYPES_OF_MEASUREMENT), dtype=np.uint32)

    # Files should only differ by seed
    rare_filter = get_rare_filter(methods)
    filter_stopwords = perform_stopword_filtering(methods)

    for i, f in enumerate(files):
        vocabulary_results[i] = sample_from_vocabulary(v(f), samples, sample_size, rare_filter=rare_filter, stop_filter=filter_stopwords)

    sample_result = np.mean(vocabulary_results, axis=0)

    with open(sample_filename, 'wb') as f:
        pickle.dump(sample_result, f)

    return sample_result
