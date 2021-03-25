import pytest
from preprocess import vocabulary
import os

def test_length():
    vi = vocabulary.VocabInfo
    # Test VocabInfo

    test_v = vi()

    # Test zero elements
    assert not len(test_v)

    # Test adding one element
    test_v['word1']
    assert len(test_v) == 1

    # Test adding previous element
    test_v['word1']
    assert len(test_v) == 1

    # Test adding another element on top of previous
    test_v['word2']
    assert len(test_v) == 2

    test_v = vi()
    for i in range(1000):
        test_v[f'word{i}']

    # Test many values
    assert len(test_v) == 1000

    # Test HashVocabInfo
    hvi = vocabulary.HashVocabInfo
    size = 1

    test_v = hvi(size)

    # Test zero elements
    assert not len(test_v)

    # Test adding one element
    test_v['word1']
    assert len(test_v) == 1

    # Test adding previous element
    test_v['word1']
    assert len(test_v) == 1

    # Test adding additional element
    test_v['word2']
    assert len(test_v) == 1

    # Test many elements
    test_v = hvi(size)
    for i in range(1000):
        test_v[f'word{i}']

    assert len(test_v) == size

    # Test different size
    size = 30
    test_v = hvi(30)
    for i in range(1000):
        test_v[f'word{i}']

    assert len(test_v) == size

def test_name_generation():
    cvf = vocabulary.create_vocabulary_filename
    vocab_dir = vocabulary.vocab_dir

    # Test single method
    corpus = 'amazon'
    methods = ['lc']
    seed = 0
    expected_filename = 'amazon_lower_vocabulary.pickle'
    assert cvf(corpus, methods, seed) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test multiple methods
    methods = ['lc', 'np']
    expected_filename = 'amazon_lower_nopunct_vocabulary.pickle'
    assert cvf(corpus, methods, seed) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test different corpus
    corpus = 'reddit'
    expected_filename = 'reddit_lower_nopunct_vocabulary.pickle'
    assert cvf(corpus, methods, seed) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test hashing
    methods = ['lc', 'np', 'h50']
    expected_filename = 'reddit_lower_nopunct_h50_s0_vocabulary.pickle'
    assert cvf(corpus, methods, seed) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test different seed
    seed = 10
    expected_filename = 'reddit_lower_nopunct_h50_s10_vocabulary.pickle'
    assert cvf(corpus, methods, seed) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        cvf(0, methods, 0)

    with pytest.raises(TypeError):
        cvf(list(), methods, 0)

    with pytest.raises(TypeError):
        cvf(set(), methods, 0)

    with pytest.raises(TypeError):
        cvf(dict(), methods, 0)

    with pytest.raises(TypeError):
        cvf(tuple((0, 0)), methods, 0)

    # Test non-list values for methods
    with pytest.raises(TypeError):
        cvf(corpus, int(), 0)

    with pytest.raises(TypeError):
        cvf(corpus, set(), 0)

    with pytest.raises(TypeError):
        cvf(corpus, str(), 0)

    with pytest.raises(TypeError):
        cvf(corpus, dict(), 0)

    with pytest.raises(TypeError):
        cvf(corpus, tuple((0, 0)), 0)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        cvf(corpus, methods, list())

    with pytest.raises(TypeError):
        cvf(corpus, methods, set())

    with pytest.raises(TypeError):
        cvf(corpus, methods, dict())

    with pytest.raises(TypeError):
        cvf(corpus, methods, tuple((0, 0)))

    with pytest.raises(TypeError):
        cvf(corpus, methods, str())

def test_text_retrieval():
    rt = vocabulary.retrieve_text

    # Test normal operation
    s = '{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}\n'
    expected = 'Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!'

    assert rt(s, True) == expected

    # Test is_json boolean
    assert rt(s, False) == s

    # Test plaintext
    assert rt(expected, False) == expected

    # Test invalid json_tag
    with pytest.raises(KeyError):
        rt(s, True, tag='clearlyFalse')

    # Test json decoding of plaintext
    import json
    with pytest.raises(json.JSONDecodeError):
        rt(expected, True)

    # Test empty string
    with pytest.raises(AttributeError):
        rt('', True)

    with pytest.raises(AttributeError):
        rt(s, True, tag='')

    # Test non-string values
    with pytest.raises(TypeError):
        rt(int(0), True)

    with pytest.raises(TypeError):
        rt(int(0), False)

    with pytest.raises(TypeError):
        rt(dict(), True)

    with pytest.raises(TypeError):
        rt(dict(), False)

    with pytest.raises(TypeError):
        rt(list(), True)

    with pytest.raises(TypeError):
        rt(list(), False)

    with pytest.raises(TypeError):
        rt(set(), True)

    with pytest.raises(TypeError):
        rt(set(), False)

    # Test non-boolean values
    with pytest.raises(TypeError):
        rt(s, int(0))

    with pytest.raises(TypeError):
        rt(s, list())

    with pytest.raises(TypeError):
        rt(s, dict())

    with pytest.raises(TypeError):
        rt(s, set())

    # Test non-string tag values
    with pytest.raises(TypeError):
        rt(s, True, tag=int(0))

    with pytest.raises(TypeError):
        rt(s, False, tag=int(0))

    with pytest.raises(TypeError):
        rt(s, True, tag=set())

    with pytest.raises(TypeError):
        rt(s, False, tag=set())

    with pytest.raises(TypeError):
        rt(s, True, tag=list())

    with pytest.raises(TypeError):
        rt(s, False, tag=list())

    with pytest.raises(TypeError):
        rt(s, True, tag=dict())

    with pytest.raises(TypeError):
        rt(s, False, tag=dict())

def test_stopword_list_generation():
    gsl = vocabulary.generate_stopword_list

    # Test normal operation
    stopword_filepath = os.path.join(os.getenv('HOME'), 'preprocess/utilities/english.txt')
    with open(stopword_filepath, 'r') as f:
        words = { w.strip('\n') for w in f.readlines() }

    stopword_list = gsl()
    assert type(stopword_list) is set
    for w in stopword_list: assert type(w) is str
    assert words == stopword_list

def test_json_check():
    cj = vocabulary.check_json

    # Test normal filepath
    assert cj('/start/home/filepath/something.json')

    # Test word only
    assert cj('json')

    # Test partial word
    assert not cj('jso')

    # Test empty string
    with pytest.raises(AttributeError):
        cj('')

    # Test non-string values
    with pytest.raises(TypeError):
        cj(int(0))

    with pytest.raises(TypeError):
        cj(list())

    with pytest.raises(TypeError):
        cj(set())

    with pytest.raises(TypeError):
        cj(dict())

def test_hashed_corpus_info_class():
    # Test base hashing functionality
    hvi = vocabulary.HashVocabInfo(2)

    for i in range(100):
        word_index = hvi[f'word{i}']
        assert word_index < 2 and word_index >= 0
        assert word_index != 2
        assert word_index > -1

    # Test different hashing number
    hvi = vocabulary.HashVocabInfo(10)

    for i in range(100):
        word_index = hvi[f'word{i}']
        assert word_index < 10 and word_index >= 0
        assert word_index != 10
        assert word_index > -1

    # Test hash value coverage

    # Want the results to be reproducible and not fail in exceptional circumstances
    import os
    os.environ['PYTHONHASHSEED'] = str(0)

    hvi = vocabulary.HashVocabInfo(2)
    testlist = set()
    for i in range(100):
        testlist.update([hvi[f'word{i}']])

    for i in range(10):
        if i < 2:
            assert i in testlist
        else:
            assert i not in testlist

    hvi = vocabulary.HashVocabInfo(10)
    testlist = set()
    for i in range(100):
        testlist.update([hvi[f'word{i}']])

    for i in range(20):
        if i < 10:
            assert i in testlist
        else:
            assert i not in testlist

def test_accurate_pickle_retrieval():
    rpv = vocabulary.retrieve_pickled_vocabulary

    # Test base functionality
    test_vocabulary = { 'word' : (1, 1), 'bacon' : (1, 2) }
    test_vocabulary_filename = 'test_vocab.pickle'

    import pickle
    with open(test_vocabulary_filename, 'wb') as f:
        pickle.dump(test_vocabulary, f)

    p = rpv(test_vocabulary_filename)
    for k, v in p.items(): assert v == test_vocabulary[k]

    # Test non-string values for vocab_filename
    with pytest.raises(TypeError):
        rpv(list())

    with pytest.raises(TypeError):
        rpv(set())

    with pytest.raises(TypeError):
        rpv(dict())

    with pytest.raises(TypeError):
        rpv(int())

    with pytest.raises(TypeError):
        rpv(tuple((0, 0)))

    os.remove(test_vocabulary_filename)
    assert not os.path.isfile(test_vocabulary_filename)

    # Test nonexistent files for vocab_filename
    assert not os.path.exists('fakefile.pickle')
    with pytest.raises(AttributeError):
        rpv('fakefile.pickle')

    # Test using relative file pathing
    assert not os.path.exists('../fakefile.pickle')
    with pytest.raises(AttributeError):
        rpv('../fakefile.pickle')

def test_corpus_info_class():
    vi = vocabulary.VocabInfo()

    # Test base functionality
    try:
        assert vi['word1'] == 0
        assert vi['word2'] == 1
        assert vi['word3'] == 2
    except:
        pytest.fail('Unexpected error')

    # Test proper data structure
    assert type(vi.dictionary) is dict

    # Test incrementing term frequency
    vi.increment_term_frequency('word1')
    assert vi.term_frequency('word1') == 1
    vi.increment_term_frequency('word1')
    assert vi.term_frequency('word1') == 2

    # Test incrementing document frequency
    vi.increment_doc_frequency('word1')
    assert vi.doc_frequency('word1') == 1
    vi.increment_doc_frequency('word1')
    assert vi.doc_frequency('word1') == 2

    # Test incrementing term frequency on unknown word
    vi.increment_term_frequency('word4')
    assert vi.term_frequency('word4') == 1

    # Test incrementing document frequency on unknown word
    vi.increment_doc_frequency('word5')
    assert vi.doc_frequency('word5') == 1

    # Test non-string keys
    with pytest.raises(TypeError):
        vi[int()]

    with pytest.raises(TypeError):
        vi[tuple((0, 0))]

    with pytest.raises(TypeError):
        vi[dict()]

    with pytest.raises(TypeError):
        vi[list()]


def test_hashing_detection():
    ph = vocabulary.perform_hashing

    # Test base performance
    methods = ['h2', 'lc']
    assert ph(methods)

    # Test larger integer
    methods = ['h300', 'lc', 'ws']
    assert ph(methods)

    # Test spatial consistency
    methods = ['lc', 'ws', 'h300']
    assert ph(methods)

    # Test bigger number
    methods = ['lc', 'ws', 'h328433']
    assert ph(methods)

    # Test no hashing
    methods = ['lc', 'ws']
    assert not ph(methods)

    # Test incorrectly formatted hash label
    methods = ['lc', 'ws', 'hs3']
    assert not ph(methods)

    # Test non-list values
    with pytest.raises(TypeError):
        ph(0)

    with pytest.raises(TypeError):
        ph(set())

    with pytest.raises(TypeError):
        ph(dict())

    with pytest.raises(TypeError):
        ph(tuple((0, 0)))

    with pytest.raises(TypeError):
        ph(str())

def test_vocabulary_directory_creation():
    from preprocess import argmanager

    rv = vocabulary.retrieve_vocabulary
    vc = argmanager.valid_corpora
    vocab_dir = os.path.join(os.getenv('HOME'), '.preprocess/vocabulary')
    vocab_filename = os.path.join(vocab_dir, 'notacorpus/notacorpus_vocabulary.pickle')
    assert not os.path.isdir(vocab_dir)
    assert not os.path.isfile(vocab_filename)
    for corpus in vc:
        assert not os.path.isdir(os.path.join(vocab_dir, f'{corpus}'))

    notacorpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/notacorpus')
    notacorpus_file = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')
    assert not os.path.isdir(notacorpus_dir)
    assert not os.path.isfile(notacorpus_file)
    os.mkdir(notacorpus_dir)

    import gzip

    s = 'word1 word2 word3 word1'
    with gzip.open(notacorpus_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    # Test normal functionality
    seed = 0
    methods = []
    corpus = 'notacorpus'

    try:
        # Twice because it should retrieve the pickled file the second time
        v = rv(corpus, methods, seed)
        v = rv(corpus, methods, seed)

    except:
        pytest.fail('Unexpected error')

    assert os.path.isdir(vocab_dir)
    for corpus in vc:
        assert os.path.isdir(os.path.join(vocab_dir, f'{corpus}'))

    os.remove(vocab_filename)
    os.remove(notacorpus_file)
    os.rmdir(notacorpus_dir)

    for corpus in vc:
        os.rmdir(os.path.join(vocab_dir, f'{corpus}'))

    os.rmdir(vocab_dir)

    assert not os.path.isfile(vocab_filename)
    assert not os.path.isfile(notacorpus_file)
    assert not os.path.isdir(notacorpus_dir)
    assert not os.path.isdir(vocab_dir)

    for corpus in vc:
        assert not os.path.isdir(os.path.join(vocab_dir, f'{corpus}'))

def test_json_vocabulary_generation():
    gv = vocabulary.generate_vocabulary
    testjson_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/testjson')
    testjson_file = os.path.join(testjson_dir, 'testjson.json.gz')
    vocab_filename = 'test_vocabulary.pickle'
    assert not os.path.isfile(vocab_filename)
    os.mkdir(testjson_dir)

    import gzip

    # Test single line
    s = '{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"word1 word2 word3 word1\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}'
    with gzip.open(testjson_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    v = gv('testjson', [], vocab_filename)
    expected = { 'word1' : [0, 2, 1], 'word2' : [1, 1, 1], 'word3' : [2, 1, 1] }

    assert os.path.isfile(vocab_filename)
    # The word index may be different depending on how the internal set reorders the words
    for key, val in v.dictionary.items(): assert val[1:] == expected[key][1:]

    os.remove(vocab_filename)
    assert not os.path.isfile(vocab_filename)

    # Test multiple lines
    s = f'{s}\n{s}\n{s}'
    print('S:', s)
    with gzip.open(testjson_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    v = gv('testjson', [], vocab_filename)
    expected = { 'word1' : [0, 6, 3], 'word2' : [1, 3, 3], 'word3' : [2, 3, 3] }

    assert os.path.isfile(vocab_filename)
    for key, val in v.dictionary.items(): assert val[1:] == expected[key][1:]

    os.remove(vocab_filename)
    os.remove(testjson_file)
    os.rmdir(testjson_dir)
    assert not os.path.isfile(vocab_filename)
    assert not os.path.isfile(testjson_file)
    assert not os.path.exists(testjson_dir)

    # Test for reddit json
    s = """{"body":"word1 word2 word3 word1","can_gild":true,"score":0}"""
    testjson_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/testredditjson')
    testjson_file = os.path.join(testjson_dir, 'testredditjson.json.gz')

    os.mkdir(testjson_dir)

    with gzip.open(testjson_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    v = gv('testredditjson', [], vocab_filename)
    expected = { 'word1' : [0, 2, 1], 'word2' : [1, 1, 1], 'word3' : [2, 1, 1] }

    assert os.path.isfile(vocab_filename)
    for key, val in v.dictionary.items(): assert val[1:] == expected[key][1:]

    os.remove(vocab_filename)
    os.remove(testjson_file)
    os.rmdir(testjson_dir)
    assert not os.path.isfile(vocab_filename)
    assert not os.path.isfile(testjson_file)
    assert not os.path.exists(testjson_dir)

def test_vocabulary_generation():
    gv = vocabulary.generate_vocabulary

    notacorpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/notacorpus')
    notacorpus_file = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')
    vocab_filename = 'test_vocabulary.pickle'
    assert not os.path.isfile(vocab_filename)
    os.mkdir(notacorpus_dir)

    import gzip

    # Test single line
    s = 'word1 word2 word3 word1'
    with gzip.open(notacorpus_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    v = gv('notacorpus', [], vocab_filename)
    expected = { 'word1' : [0, 2, 1], 'word2' : [1, 1, 1], 'word3' : [2, 1, 1] }

    assert os.path.isfile(vocab_filename)
    # The word index may be different depending on how the internal set reorders the words
    for key, val in v.dictionary.items(): assert val[1:] == expected[key][1:]

    os.remove(vocab_filename)
    assert not os.path.isfile(vocab_filename)

    # Test multiple lines
    s = 'word1 word2 word3 word1\nword1 word2 word3 word1\nword1 word2 word3 word1'
    with gzip.open(notacorpus_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    v = gv('notacorpus', [], vocab_filename)
    expected = { 'word1' : [0, 6, 3], 'word2' : [1, 3, 3], 'word3' : [2, 3, 3] }

    assert os.path.isfile(vocab_filename)
    for key, val in v.dictionary.items(): assert val[1:] == expected[key][1:]

    os.remove(vocab_filename)
    os.remove(notacorpus_file)
    os.rmdir(notacorpus_dir)
    assert not os.path.isfile(vocab_filename)
    assert not os.path.isfile(notacorpus_file)
    assert not os.path.exists(notacorpus_dir)

    # Test non-existent corpus
    with pytest.raises(ValueError):
        gv('notacorpus', [], vocab_filename)

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        gv(int(), [], vocab_filename)

    with pytest.raises(TypeError):
        gv(set(), [], vocab_filename)

    with pytest.raises(TypeError):
        gv(list(), [], vocab_filename)

    with pytest.raises(TypeError):
        gv(dict(), [], vocab_filename)

    with pytest.raises(TypeError):
        gv(tuple((0, 0)), [], vocab_filename)

    # Test non-list values for methods
    with pytest.raises(TypeError):
        gv('notacorpus', set(), vocab_filename)

    with pytest.raises(TypeError):
        gv('notacorpus', dict(), vocab_filename)

    with pytest.raises(TypeError):
        gv('notacorpus', str(), vocab_filename)

    with pytest.raises(TypeError):
        gv('notacorpus', tuple((0, 0)), vocab_filename)

    # Test non-string values for vocab_filename
    with pytest.raises(TypeError):
        gv('notacorpus', [], int())

    with pytest.raises(TypeError):
        gv('notacorpus', [], set())

    with pytest.raises(TypeError):
        gv('notacorpus', [], list())

    with pytest.raises(TypeError):
        gv('notacorpus', [], dict())

    # Test empty string for vocab_filename
    with pytest.raises(AttributeError):
        gv('notacorpus', [], '')

def test_create_bpe_set():
    cbs = vocabulary.create_bpe_set

    corpus = 'testamazon'
    methods = []
    is_json = True
    tag = 'reviewText'

    # Test various vocab sizes
    vocab_size = 100
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=False)
    assert vocab_size == len(bpe_set)

    vocab_size = 102
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=False)
    assert vocab_size == len(bpe_set)

    vocab_size = 113
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=False)
    assert vocab_size == len(bpe_set)

    vocab_size = 157
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=False)
    assert vocab_size == len(bpe_set)

    notacorpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/notacorpus')
    notacorpus_file = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')
    os.mkdir(notacorpus_dir)

    import gzip

    s = 'tord1 word1 world2 work3 wart1 towt'
    with gzip.open(notacorpus_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    is_json = False
    corpus = 'notacorpus'

    vocab_size = 12
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=False)

    assert len(bpe_set) == vocab_size
    assert bpe_set == set({'or', 'w', 'r', 'l', 'd', 'k', '1', '2', '3', 't', 'a', 'o'})

    s = ''
    for i in range(100):
        s += f'{i} '

    with gzip.open(notacorpus_file, 'wb') as f:
        f.write(s.encode('utf-8'))

    # Test reduce_vocab option
    vocab_size = 95
    bpe_set = cbs(corpus, methods, is_json, tag, vocab_size, reduce_vocab=True)
    assert len(bpe_set) == vocab_size

    os.remove(notacorpus_file)
    os.rmdir(notacorpus_dir)
    assert not os.path.isfile(notacorpus_file)
    assert not os.path.exists(notacorpus_dir)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        cbs('', [], is_json, tag, vocab_size, reduce_vocab=False)

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        cbs(int(), [], is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs(set(), [], is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs(list(), [], is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs(dict(), [], is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs(tuple((0, 0)), [], is_json, tag, vocab_size, reduce_vocab=False)

    # Test non-list values for methods
    with pytest.raises(TypeError):
        cbs('notacorpus', set(), is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', dict(), is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', str(), is_json, tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', tuple((0, 0)), is_json, tag, vocab_size, reduce_vocab=False)

    # Test non-bool values for is_json
    with pytest.raises(TypeError):
        cbs('notacorpus', [], set(), tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], dict(), tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], str(), tag, vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], tuple((0, 0)), tag, vocab_size, reduce_vocab=False)

    # Test non-string values for tag
    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, set(), vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, dict(), vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, tuple((0, 0)), vocab_size, reduce_vocab=False)

    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, bool(True), vocab_size, reduce_vocab=False)

    # Test non-int values for vocab_size
    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, tag, set())

    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, tag, dict())

    with pytest.raises(TypeError):
        cbs('notacorpus', [], is_json, tag, tuple((0, 0)))

def test_create_bpe_set_filename():
    cbsf = vocabulary.create_bpe_set_filename
    vocab_dir = vocabulary.vocab_dir

    # Test single method
    corpus = 'amazon'
    methods = ['lc']
    seed = 0
    expected_filename = 'amazon_lower_bpe_set.pickle'
    assert cbsf(corpus, methods) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test multiple methods
    methods = ['lc', 'np']
    expected_filename = 'amazon_lower_nopunct_bpe_set.pickle'
    assert cbsf(corpus, methods) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test different corpus
    corpus = 'reddit'
    expected_filename = 'reddit_lower_nopunct_bpe_set.pickle'
    assert cbsf(corpus, methods) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test hashing
    methods = ['lc', 'np', 'h50']
    expected_filename = 'reddit_lower_nopunct_h50_bpe_set.pickle'
    assert cbsf(corpus, methods) == os.path.join(vocab_dir, f'{corpus}/{expected_filename}')

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        cbsf(0, methods)

    with pytest.raises(TypeError):
        cbsf(list(), methods)

    with pytest.raises(TypeError):
        cbsf(set(), methods)

    with pytest.raises(TypeError):
        cbsf(dict(), methods)

    with pytest.raises(TypeError):
        cbsf(tuple((0, 0)), methods)

    # Test non-list values for methods
    with pytest.raises(TypeError):
        cbsf(corpus, int())

    with pytest.raises(TypeError):
        cbsf(corpus, set())

    with pytest.raises(TypeError):
        cbsf(corpus, str())

    with pytest.raises(TypeError):
        cbsf(corpus, dict())

    with pytest.raises(TypeError):
        cbsf(corpus, tuple((0, 0)))
