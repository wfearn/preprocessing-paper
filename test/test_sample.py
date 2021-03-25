import pytest
import os
import preprocess
from preprocess import sample
from preprocess import vocabulary

def test_sample_function():
    import pickle
    import numpy as np
    from preprocess import argmanager

    s = sample.sample
    sample_dir = os.path.join(os.getenv('HOME'), '.preprocess/samples')
    vocab_dir = os.path.join(os.getenv('HOME'), '.preprocess/vocabulary')
    notacorpus_vocab_dir = os.path.join(vocab_dir, 'notacorpus')
    assert not os.path.isdir(sample_dir)
    assert not os.path.isdir(vocab_dir)
    for c in argmanager.valid_corpora:
        assert not os.path.isdir(os.path.join(sample_dir, c))

    try:
        corpus = 'notacorpus'
        methods = []
        sample_filename = sample.create_sample_filename(corpus, methods)

        v = { 'word1' : [0, 2, 1], 'word2' : [1, 2, 1] }
        vocab = vocabulary.VocabInfo()
        vocab.dictionary = v

        # fake seed of 0
        os.mkdir(vocab_dir)
        os.mkdir(notacorpus_vocab_dir)
        vocab_filename = vocabulary.create_vocabulary_filename(corpus, methods, 0)
        with open(vocab_filename, 'wb') as f:
            pickle.dump(vocab, f)

        # Test directory creation
        assert not os.path.isfile(sample_filename)

        one_sample = s(corpus, methods)
        assert os.path.isdir(sample_dir)
        for c in argmanager.valid_corpora:
            assert os.path.isdir(os.path.join(sample_dir, c))

        assert os.path.isfile(sample_filename)

        # Test correct object shape
        assert len(one_sample.shape) == 2

        # Test pickle retrieval
        test_sample = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4]])
        with open(sample_filename, 'wb') as f:
            pickle.dump(test_sample, f)

        one_sample = s(corpus, methods)

        assert test_sample.shape == one_sample.shape
        for i, val in enumerate(one_sample):
            assert set(val) == set(test_sample[i])

    except:
        pytest.fail('Unexpected exception')

    finally:
        os.remove(sample_filename)
        os.remove(vocab_filename)
        for c in argmanager.valid_corpora:
            os.rmdir(os.path.join(sample_dir, c))
        os.rmdir(sample_dir)
        os.rmdir(notacorpus_vocab_dir)
        os.rmdir(vocab_dir)

        assert not os.path.isfile(sample_filename)
        assert not os.path.isfile(vocab_filename)
        for c in argmanager.valid_corpora:
            assert not os.path.isdir(os.path.join(sample_dir, c))
        assert not os.path.isdir(sample_dir)
        assert not os.path.isdir(vocab_dir)
        assert not os.path.isdir(notacorpus_vocab_dir)

def test_sample_name_generation():
    csf = sample.create_sample_filename

    sample_dir = os.path.join(os.getenv('HOME'), '.preprocess/samples')
    corpus = 'notacorpus'
    methods = []

    # Test normal functionailty
    expected = os.path.join(sample_dir, 'notacorpus/notacorpus_sample.pickle')
    assert csf(corpus, methods) == expected

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        csf(int(), methods)

    with pytest.raises(TypeError):
        csf(dict(), methods)

    with pytest.raises(TypeError):
        csf(set(), methods)

    with pytest.raises(TypeError):
        csf(tuple((0, 0)), methods)

    with pytest.raises(TypeError):
        csf(list(), methods)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        csf('', methods)

    # Test non-list values for methods
    with pytest.raises(TypeError):
        csf(corpus, int())

    with pytest.raises(TypeError):
        csf(corpus, set())

    with pytest.raises(TypeError):
        csf(corpus, dict())

    with pytest.raises(TypeError):
        csf(corpus, tuple((0, 0)))

def test_sample_dir_creation():

    csd = sample.create_sample_dirs
    sample_dir = './sample'

    # Test normal functionality
    assert not os.path.isdir(sample_dir)

    csd(sample_dir)

    assert os.path.isdir(sample_dir)

    for valid_corpus in preprocess.argmanager.valid_corpora:
        corpus_dir = os.path.join(sample_dir, valid_corpus)
        assert os.path.isdir(corpus_dir)
        os.rmdir(corpus_dir)
        assert not os.path.isdir(corpus_dir)

    os.rmdir(sample_dir)
    assert not os.path.isdir(sample_dir)

    # Test non-string values for sample_dir
    with pytest.raises(TypeError):
        csd(int())

    with pytest.raises(TypeError):
        csd(dict())

    with pytest.raises(TypeError):
        csd(list())

    with pytest.raises(TypeError):
        csd(tuple((0, 0)))

    with pytest.raises(TypeError):
        csd(set())

    # Test empty string value for sample_dir
    with pytest.raises(AttributeError):
        csd('')

def test_pickle_retrieval():
    rsp = sample.retrieve_sample_pickle

    testpickle_filename = 'testpickle.pickle'
    testvalue = {1, 2, 3, 4, 5}

    import pickle
    with open(testpickle_filename, 'wb') as f:
        pickle.dump(testvalue, f)

    # Test normal functionality
    p = rsp(testpickle_filename)
    assert p == testvalue

    os.remove(testpickle_filename)
    assert not os.path.isfile(testpickle_filename)

    # Test nonexistant file for sampple_filename
    with pytest.raises(AttributeError):
        rsp(testpickle_filename)

    # Test non-string values for sample_filename
    with pytest.raises(TypeError):
        rsp(int())

    with pytest.raises(TypeError):
        rsp(dict())

    with pytest.raises(TypeError):
        rsp(list())

    with pytest.raises(TypeError):
        rsp(set())

    with pytest.raises(TypeError):
        rsp(tuple((0, 0)))

    # Test empty string for sample_filename
    with pytest.raises(AttributeError):
        rsp('')

def test_get_rare_filter():
    grf = sample.get_rare_filter

    # Test normal functionality
    assert grf(['r300']) == 300

    # Test single digit
    assert grf(['r1']) == 1

    # Test no digit
    assert grf(['r']) == 1

    # Test longer digits
    assert grf(['r123456']) == 123456

    # Test no rare filter at all
    assert grf(['lc']) == 1

    # Test negative value
    # Doesn't get matched by the regex
    assert grf(['r-3000']) == 1

    # Test zero value
    with pytest.raises(AttributeError):
        grf(['r0'])

    # Test empty list
    assert grf([]) == 1

    # Test non-list values for vocab_file
    with pytest.raises(TypeError):
        grf(int())

    with pytest.raises(TypeError):
        grf(dict())

    with pytest.raises(TypeError):
        grf(set())

    with pytest.raises(TypeError):
        grf(str())

    with pytest.raises(TypeError):
        grf(tuple((0, 0)))

def test_categorical_creation():

    cc = sample.create_categorical

    # Test base functionality
    vocab = { 'word1' : [0, 2, 1], 'word2' : [1, 2, 1] }
    v = vocabulary.VocabInfo()
    v.dictionary = vocab
    expected_p_values = [.5, .5]
    expected_sample_values = ['word1', 'word2']

    sample_values, p_values = cc(v)
    assert expected_sample_values == sample_values
    for i, val in enumerate(expected_p_values):
        assert val == p_values[i]

    # Test different ratios
    vocab = { 'word1' : [0, 1, 1], 'word2' : [1, 3, 1] }
    v.dictionary = vocab

    expected_p_values = [.25, .75]
    sample_values, p_values = cc(v)

    assert sample_values == expected_sample_values
    for i, val in enumerate(expected_p_values):
        assert val == p_values[i]

    # Test more token types
    vocab = { 'word1' : [0, 1, 1], 'word2' : [1, 3, 1], 'word3' : [2, 4, 1] }
    v.dictionary = vocab

    expected_p_values = [.125, .375, .5]
    expected_sample_values = ['word1', 'word2', 'word3']
    sample_values, p_values = cc(v)

    assert sample_values == expected_sample_values
    for i, val in enumerate(expected_p_values):
        assert val == p_values[i]

    # Test with HashVocabInfo
    v = vocabulary.HashVocabInfo(6)
    v.dictionary = vocab

    expected_p_values = [.125, .375, .5]
    expected_sample_values = ['word1', 'word2', 'word3']
    sample_values, p_values = cc(v)

    assert sample_values == expected_sample_values
    for i, val in enumerate(expected_p_values):
        assert val == p_values[i]

    # Test non-VocabInfo types
    with pytest.raises(TypeError):
        cc(str())

    with pytest.raises(TypeError):
        cc(list())

    with pytest.raises(TypeError):
        cc(dict())

    with pytest.raises(TypeError):
        cc(set())

    with pytest.raises(TypeError):
        cc(tuple((0, 0)))

    with pytest.raises(TypeError):
        cc(int())

def test_sample_from_vocabulary():
    from preprocess import sample

    sfv = sample.sample_from_vocabulary
    type_index = sample.TYPE_INDEX
    token_index = sample.TOKEN_INDEX
    types_of_measurement = sample.TYPES_OF_MEASUREMENT
    step_size = sample.DEFAULT_STEP_SIZE

    vocab = { 'word1' : [0, 2, 1], 'word2' : [1, 2, 1] }
    v = vocabulary.VocabInfo()
    v.dictionary = vocab
    num_samples = 5
    sample_size = 2000
    s = sfv(v, num_samples, sample_size)
    num_measurements = (sample_size // step_size) + 1

    # Test correct shape
    assert s.shape == (num_measurements, types_of_measurement)

    # Test correst number of tokens sampled
    assert s[-1][token_index] == sample_size + 1

    # Test correct number of types
    assert s[-1][type_index] == len(vocab)

    # Test default non-stopword removal
    stopword_list = list(vocabulary.generate_stopword_list())
    vocab = { stopword_list[0] : [0, 2, 1], 'word2' : [1, 2, 1] }
    v = vocabulary.VocabInfo()
    v.dictionary = vocab
    s = sfv(v, num_samples, sample_size)

    # Test correct number of types without stopword filtering
    assert s[-1][type_index] == len(vocab)

    s = sfv(v, num_samples, sample_size, stop_filter=True)

    # Test correct number of types with stopword filtering
    assert s[-1][type_index] == (len(vocab) - 1)

    vocab = { 'word1' : [0, 2, 1], 'word2' : [1, 2, 2] }
    v = vocabulary.VocabInfo()
    v.dictionary = vocab
    s = sfv(v, num_samples, sample_size, rare_filter=2)

    # Test correct number of types with raised rare word filter
    assert s[-1][type_index] == (len(vocab) - 1)

    vocab = { 'word1' : [0, 2, 1], 'word2' : [1, 2, 1] }
    v = vocabulary.VocabInfo()
    v.dictionary = vocab
    s = sfv(v, num_samples, sample_size, rare_filter=2)

    # Test all words filtered when sufficiently high rare word filter
    assert not s[-1][type_index] and s[-1][token_index]

    # Test non-VocabInfo types
    with pytest.raises(TypeError):
        sfv(int(), num_samples, sample_size, rare_filter=1)

    with pytest.raises(TypeError):
        sfv(dict(), num_samples, sample_size, rare_filter=1)

    with pytest.raises(TypeError):
        sfv(set(), num_samples, sample_size, rare_filter=1)

    with pytest.raises(TypeError):
        sfv(tuple((0, 0)), num_samples, sample_size, rare_filter=1)

    with pytest.raises(TypeError):
        sfv(list(), num_samples, sample_size, rare_filter=1)

    # Test non-int type for num_samples
    with pytest.raises(TypeError):
        sfv(v, set(), sample_size)

    with pytest.raises(TypeError):
        sfv(v, list(), sample_size)

    with pytest.raises(TypeError):
        sfv(v, dict(), sample_size)

    with pytest.raises(TypeError):
        sfv(v, tuple((0, 0)), sample_size)

    # Test non-int type for sample_size
    with pytest.raises(TypeError):
        sfv(v, num_samples, list())

    with pytest.raises(TypeError):
        sfv(v, num_samples, set())

    with pytest.raises(TypeError):
        sfv(v, num_samples, dict())

    with pytest.raises(TypeError):
        sfv(v, num_samples, tuple((0, 0)))

    # Test non-bool type for stop_filter
    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, stop_filter=int())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, stop_filter=dict())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, stop_filter=list())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, stop_filter=tuple((0, 0)))

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, stop_filter=set())

    # Test non-int value for rare_filter
    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, rare_filter=list())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, rare_filter=set())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, rare_filter=dict())

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, rare_filter=tuple((0, 0)))

    with pytest.raises(TypeError):
        sfv(v, num_samples, sample_size, rare_filter=bool())

    # Test 0 value for rare filter
    with pytest.raises(AttributeError):
        sfv(v, num_samples, sample_size, rare_filter=0)

    # Test negative value for rare filter
    with pytest.raises(AttributeError):
        sfv(v, num_samples, sample_size, rare_filter=-1)

    with pytest.raises(AttributeError):
        sfv(v, num_samples, sample_size, rare_filter=-100000)

    # Test negative value for num_samples
    with pytest.raises(AttributeError):
        sfv(v, -1, sample_size)

    with pytest.raises(AttributeError):
        sfv(v, -10000, sample_size)

    # Test negative values for sample_size
    with pytest.raises(AttributeError):
        sfv(v, num_samples, -1)

    with pytest.raises(AttributeError):
        sfv(v, num_samples, -10000)

def test_relevant_sample_file_retrieval():
    from pathlib import Path

    corpus = 'notacorpus'
    methods = []

    grsf = sample.get_relevant_sample_files
    vocab_dir = preprocess.vocabulary.vocab_dir
    notacorpus_dir = os.path.join(vocab_dir, corpus)

    assert not os.path.isdir(vocab_dir)
    assert not os.path.isdir(notacorpus_dir)
    os.mkdir(vocab_dir)
    os.mkdir(notacorpus_dir)


    # Test normal usage
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert expected_files == grsf(corpus, methods)

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test with basic seed and method
    methods = ['h10']
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s0_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert expected_files == grsf(corpus, methods)

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test multiple seeds
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s0_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s1_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s2_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert expected_files == grsf(corpus, methods)

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test seed without hash
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_s0_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert not grsf(corpus, methods)

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test hash without seed
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_h10_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert not grsf(corpus, methods)

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test seed without hash among valid names
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_s0_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s0_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s1_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s2_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert expected_files - grsf(corpus, methods) == { os.path.join(notacorpus_dir, 'notacorpus_s0_vocabulary.pickle') }

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    # Test retrieving up to 10 seed files
    expected_files = {
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s0_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s1_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s2_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s3_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s4_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s5_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s6_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s7_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s8_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s9_vocabulary.pickle'),
                        os.path.join(notacorpus_dir, 'notacorpus_h10_s10_vocabulary.pickle')
                     }

    for f in expected_files:
        Path(f).touch()

    assert expected_files - grsf(corpus, methods) == { os.path.join(notacorpus_dir, 'notacorpus_h10_s10_vocabulary.pickle') }

    for f in expected_files:
        os.remove(f)
        assert not os.path.isfile(f)

    os.rmdir(notacorpus_dir)
    os.rmdir(vocab_dir)
    assert not os.path.isdir(notacorpus_dir)
    assert not os.path.isdir(vocab_dir)

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        grsf(int(), methods)

    with pytest.raises(TypeError):
        grsf(dict(), methods)

    with pytest.raises(TypeError):
        grsf(list(), methods)

    with pytest.raises(TypeError):
        grsf(set(), methods)

    with pytest.raises(TypeError):
        grsf(tuple((0, 0)), methods)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        grsf('', methods)

    # Test non-list value for corpus
    with pytest.raises(TypeError):
        grsf(corpus, int())

    with pytest.raises(TypeError):
        grsf(corpus, dict())

    with pytest.raises(TypeError):
        grsf(corpus, set())

    with pytest.raises(TypeError):
        grsf(corpus, tuple((0, 0)))
