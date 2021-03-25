import os
from preprocess import argmanager

def get_base_and_ext(filename):
    import pathlib
    ext = ''.join(pathlib.Path(filename).suffixes)
    return filename[:-len(ext)], ext # base is file without extensions

def check_filenames(combined_name, files):
    for f in files:
        if combined_name not in f:
            raise ValueError(f'{f} not a subset of {combined_name}')

def combine_pickles(combined_name, files):
    check_filenames(combined_name, files)

    import pickle

    _, ext = get_base_and_ext(files[0])

    aggregate_results = list()

    if os.path.isfile(combined_name):
        with open(f'{combined_name}.pickle', 'rb') as f:
            aggregate_results = pickle.load(f)

    for f in files:

        with open(f, 'rb') as pickle_file:
            results = pickle.load(pickle_file)
            aggregate_results.append(results)

        os.remove(f)

    with open(f'{combined_name}.pickle', 'wb') as f:
        pickle.dump(aggregate_results, f)

def combine_corpora(combined_name, files):
    check_filenames(combined_name, files)

    import gzip

    # Neatly accounts for .json.gz type extensions as well as others
    _, ext = get_base_and_ext(files[0])

    with gzip.open(f'{combined_name}{ext}', 'wb') as corpus:
        for f in files:

            with gzip.open(f, 'rb') as subcorpus:
                text = subcorpus.read().decode('utf-8')
                corpus.write(f'{text}'.encode('utf-8')) # is the encode/decode necessary?

            os.remove(f)

def combine(directory):

    def no_number_base(b):
        return '_'.join(b.split('_')[:-1])

    subgroups = dict()
    for filename in os.listdir(directory):
        base, ext = get_base_and_ext(filename)
        if not base.split('_')[-1].isdigit(): continue

        no_num_base = no_number_base(base)
        full_dir = os.path.join(directory, no_num_base)
        if full_dir not in subgroups.keys():
            subgroups[full_dir] = list()

        subgroups[full_dir].append(os.path.join(directory, filename))

    for groupname, group in subgroups.items():
        _, ext = os.path.splitext(group[0])
        if ext[1:] == 'pickle': # ext comes out as '.pickle'
            combine_pickles(groupname, group)
        else:
            combine_corpora(groupname, group)

def run_combiner():

    corpus = argmanager.parse_args().corpus

    corpus_dir = argmanager.corpus_dir(corpus)
    results_dir = argmanager.results_dir(corpus)

    combine(corpus_dir)
    combine(results_dir)

if __name__ == '__main__':
    run_combiner()
