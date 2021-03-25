from preprocess import methods

def retrieve_extractor(filename):
    if 'json' in filename: return methods.json_processor
    else: return methods.plaintext_processor

def preprocess_corpus(corpus, treatments, offset=None, run_number=None):
    if corpus == '':
        raise AttributeError('Invalid empty corpus name')
    if corpus is None:
        raise AttributeError('Invalid corpus NoneType')
    if offset is not None and offset <= 0:
        raise AttributeError('Invalid offset value must be positive')
    if run_number is not None and run_number < 0:
        raise AttributeError('Invalid run_number value must be nonnegative')

    from preprocess import argmanager

    import os
    import gzip

    corpus_file = argmanager.retrieve_corpus_file(corpus, treatments)
    if os.path.isfile(corpus_file):
        return gzip.open(corpus_file, 'rb')

    if offset is None or run_number is None:
        raise AttributeError('Insufficient data to retrieve corpus')


    new_corpus_file = argmanager.retrieve_corpus_file(corpus, treatments, run_number)
    base_corpus_file = argmanager.retrieve_corpus_file(corpus, [])
    text_dict = argmanager.corpus_text_dict

    preprocessor = methods.create_preprocessor(treatments)
    extractor = retrieve_extractor(new_corpus_file)(preprocessor, tag=text_dict[corpus])

    with gzip.open(new_corpus_file, 'wb') as f, gzip.open(base_corpus_file, 'rb') as g:
        start_line = run_number * offset
        end_line = start_line + offset

        for i, line in enumerate(g):
            if i < start_line: continue
            elif i >= end_line: break
            elif i >= start_line and i < end_line:
                line = line.decode('utf-8')
                f.write(f'{extractor(line)}\n'.encode('utf-8'))
