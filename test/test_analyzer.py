from preprocess import analyzer, argmanager, importer
from collections import namedtuple
import numpy as np
import preprocess
import pickle
import pytest
import os

def test_generate_analyze_filename():
    gaf = analyzer.generate_analyze_filename
    results_dir = analyzer.results_directory

    corpus = 'notacorpus'
    methods = []
    model = 'svm'
    train_size = 777
    seed = 16

    # Test base functionality
    corpus_dir = os.path.join(results_dir, corpus)
    processed_name = argmanager.processed_corpus_name(corpus, methods)
    expected = os.path.join(corpus_dir, f'{processed_name}_{model}_corpus{train_size}_results_{seed}.pickle')
    assert gaf(corpus, methods, model, train_size, seed) == expected

    # Test different model
    model = 'naive'
    expected = os.path.join(corpus_dir, f'{processed_name}_{model}_corpus{train_size}_results_{seed}.pickle')
    assert gaf(corpus, methods, model, train_size, seed) == expected

    # Test different train size
    train_size = 17173
    expected = os.path.join(corpus_dir, f'{processed_name}_{model}_corpus{train_size}_results_{seed}.pickle')
    assert gaf(corpus, methods, model, train_size, seed) == expected

    # Test different seed
    seed = 673843
    expected = os.path.join(corpus_dir, f'{processed_name}_{model}_corpus{train_size}_results_{seed}.pickle')
    assert gaf(corpus, methods, model, train_size, seed) == expected

    # Test non-string for corpus
    with pytest.raises(TypeError):
        gaf(int(), methods, model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(dict(), methods, model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(set(), methods, model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(list(), methods, model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(tuple((0, 0)), methods, model, train_size, seed)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        gaf('', methods, model, train_size, seed)

    # Test non-string for model
    with pytest.raises(TypeError):
        gaf(corpus, methods, int(), train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, dict(), train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, set(), train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, list(), train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, tuple((0, 0)), train_size, seed)

    # Test empty string for model
    with pytest.raises(AttributeError):
        gaf(corpus, methods, '', train_size, seed)

    # Test non-list value for method
    with pytest.raises(TypeError):
        gaf(corpus, int(), model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, dict(), model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, set(), model, train_size, seed)

    with pytest.raises(TypeError):
        gaf(corpus, tuple((0, 0)), model, train_size, seed)

    # Test non-int values for train_size
    with pytest.raises(TypeError):
        gaf(corpus, methods, model, set(), seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, dict(), seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, list(), seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, tuple((0, 0)), seed)

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, str(), seed)

    # Test non-positive values for train_size
    with pytest.raises(AttributeError):
        gaf(corpus, methods, model, 0, seed)

    with pytest.raises(AttributeError):
        gaf(corpus, methods, model, -1, seed)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        gaf(corpus, methods, model, train_size, set())

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, train_size, list())

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, train_size, dict())

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, train_size, tuple((0, 0)))

    with pytest.raises(TypeError):
        gaf(corpus, methods, model, train_size, str())

def test_q_directory_creation():
    cqd = analyzer.create_q_directories

    q_dir = analyzer.q_directory

    assert not os.path.isdir(q_dir)
    for c in preprocess.argmanager.valid_corpora:
        assert not os.path.isdir(os.path.join(q_dir, c))

    cqd()

    assert os.path.isdir(q_dir)
    for c in preprocess.argmanager.valid_corpora:
        assert os.path.isdir(os.path.join(q_dir, c))

    for c in preprocess.argmanager.valid_corpora:
        os.rmdir(os.path.join(q_dir, c))
        assert not os.path.isdir(os.path.join(q_dir, c))

    os.rmdir(q_dir)
    assert not os.path.isdir(q_dir)

def test_q_filename_generator():
    cqf = analyzer.create_q_filename

    corpus = 'amazon'
    methods = ['lc']
    train_size = 5000
    seed = 0

    q_dir = analyzer.q_directory

    q_amazon_dir = os.path.join(q_dir, 'amazon')
    q_reddit_dir = os.path.join(q_dir, 'reddit')

    # Test base functionality
    expected = 'amazon_lower_s0_corpus5000_Q.pickle'
    assert cqf(corpus, methods, train_size, seed) == os.path.join(q_amazon_dir, expected)

    # Test changing train_size
    train_size = 673894
    expected = 'amazon_lower_s0_corpus673894_Q.pickle'
    assert cqf(corpus, methods, train_size, seed) == os.path.join(q_amazon_dir, expected)

    # Test changing methods
    methods = ['np']
    expected = 'amazon_nopunct_s0_corpus673894_Q.pickle'
    assert cqf(corpus, methods, train_size, seed) == os.path.join(q_amazon_dir, expected)

    # Test changing seed
    seed = 6
    expected = 'amazon_nopunct_s6_corpus673894_Q.pickle'
    assert cqf(corpus, methods, train_size, seed) == os.path.join(q_amazon_dir, expected)

    # Test changing corpus
    corpus = 'reddit'
    expected = 'reddit_nopunct_s6_corpus673894_Q.pickle'
    assert cqf(corpus, methods, train_size, seed) == os.path.join(q_reddit_dir, expected)

    # Test no methods
    expected = 'reddit_s6_corpus673894_Q.pickle'
    assert cqf(corpus, [], train_size, seed) == os.path.join(q_reddit_dir, expected)

    # Test non-string for corpus
    with pytest.raises(TypeError):
        cqf(int(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cqf(dict(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cqf(set(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cqf(list(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cqf(tuple((0, 0)), methods, train_size, seed)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        cqf('', methods, train_size, seed)

    # Test non-list value for method
    with pytest.raises(TypeError):
        cqf(corpus, int(), train_size, seed)

    with pytest.raises(TypeError):
        cqf(corpus, dict(), train_size, seed)

    with pytest.raises(TypeError):
        cqf(corpus, set(), train_size, seed)

    with pytest.raises(TypeError):
        cqf(corpus, tuple((0, 0)), train_size, seed)

    # Test non-int values for train_size
    with pytest.raises(TypeError):
        cqf(corpus, methods, set(), seed)

    with pytest.raises(TypeError):
        cqf(corpus, methods, dict(), seed)

    with pytest.raises(TypeError):
        cqf(corpus, methods, list(), seed)

    with pytest.raises(TypeError):
        cqf(corpus, methods, tuple((0, 0)), seed)

    with pytest.raises(TypeError):
        cqf(corpus, methods, str(), seed)

    # Test non-positive values for train_size
    with pytest.raises(AttributeError):
        cqf(corpus, methods, 0, seed)

    with pytest.raises(AttributeError):
        cqf(corpus, methods, -1, seed)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        cqf(corpus, methods, train_size, set())

    with pytest.raises(TypeError):
        cqf(corpus, methods, train_size, list())

    with pytest.raises(TypeError):
        cqf(corpus, methods, train_size, dict())

    with pytest.raises(TypeError):
        cqf(corpus, methods, train_size, tuple((0, 0)))

    with pytest.raises(TypeError):
        cqf(corpus, methods, train_size, str())

def test_get_targets():
    gt = analyzer.get_targets

    corpus = importer.Corpus
    document = importer.Document
    documents = [document([], {'target' : 0}) for _ in range(10)]

    c = corpus(documents, [], {})

    # Testing normal functionality
    expected = [0 for _ in range(10)]
    assert gt(c) == expected

    documents = [document([], {'target' : i}) for i in range(10)]
    c = corpus(documents, [], {})

    # Test non-trivial case
    expected = list(range(10))
    assert gt(c) == expected

    # Test incorrect metadata key
    documents = [document([], {'notarget' : i}) for i in range(10)]
    c = corpus(documents, [], {})
    with pytest.raises(KeyError):
        gt(c)

    # Test incorrect document class
    document = namedtuple('document', 'tokens')
    documents = [document([]) for _ in range(10)]

    c = corpus(documents, [], {})
    with pytest.raises(AttributeError):
        gt(c)

    # Test non-metadata attribute
    document = namedtuple('document', 'tokens meetadata')
    documents = [document([], {}) for _ in range(10)]

    c = corpus(documents, [], {})
    with pytest.raises(AttributeError):
        gt(c)

def test_retrieve_q():
    rq = analyzer.retrieve_q
    q_dir = analyzer.q_directory
    nac_dir = os.path.join(q_dir, 'notacorpus')

    assert not os.path.isdir(q_dir)
    for c in argmanager.valid_corpora:
        assert not os.path.isdir(os.path.join(q_dir, c))

    corpus = 'notacorpus'
    methods = []
    train_size = 100
    seed = 10

    nac_filepath = os.path.join(nac_dir, f'{corpus}_s{seed}_corpus{train_size}_Q.pickle')
    os.mkdir(q_dir)
    os.mkdir(nac_dir)
    nac_content = [0 for _ in range(10)]
    with open(nac_filepath, 'wb') as f:
        pickle.dump(nac_content, f)

    q = rq(corpus, methods, train_size, seed)

    # Assert directory creation and remove created folders
    os.remove(nac_filepath)
    os.rmdir(nac_dir)
    os.rmdir(q_dir)
    assert not os.path.isfile(nac_filepath)
    assert not os.path.isdir(nac_dir)
    assert not os.path.isdir(q_dir)

    # Assert same pickle retrieval
    assert q == nac_content

    # Imported corpus setup
    Corpus = importer.Corpus
    Document = importer.Document
    Token = importer.Token

    train_documents = [Document([Token(0), Token(1)], {'target' : 0})]
    test_documents = [Document([Token(0), Token(1)], {'target' : 0})]

    train = Corpus(train_documents, {0, 1}, {})
    test = Corpus(train_documents, {}, {})

    icd = importer.import_corpus_dir
    nac_icd = os.path.join(icd, 'notacorpus')
    nac_icd_filepath = os.path.join(nac_icd, f'notacorpus_s{seed}_corpus{train_size}.pickle')

    os.mkdir(icd)
    os.mkdir(nac_icd)
    with open(nac_icd_filepath, 'wb') as f:
        pickle.dump((train, test), f)

    nac_q = rq(corpus, methods, train_size, seed)

    # Test directory and pickle creation
    assert os.path.isdir(q_dir)
    for valid_corpus in argmanager.valid_corpora:
        assert os.path.isdir(os.path.join(q_dir, valid_corpus))
    assert os.path.isfile(nac_filepath)

    # Test proper corpus and correct calculation
    expected = np.asarray([[0, 0.5, 1], [0.5, 0, 1]])
    assert nac_q.shape == (2, 3)
    for i in range(nac_q.shape[0]):
        for j in range(nac_q.shape[1]):
            assert nac_q[i, j] == expected[i, j]

    os.remove(nac_icd_filepath)
    os.rmdir(nac_icd)
    os.rmdir(icd)
    assert not os.path.isfile(nac_icd_filepath)
    assert not os.path.isdir(nac_icd)
    assert not os.path.isdir(icd)

    os.remove(nac_filepath)
    for valid_corpus in argmanager.valid_corpora:
        corpus_dir = os.path.join(q_dir, valid_corpus)
        os.rmdir(corpus_dir)
        assert not os.path.isdir(corpus_dir)

    os.rmdir(q_dir)
    assert not os.path.isdir(q_dir)
    assert not os.path.isfile(nac_filepath)

    # Test non-string for corpus
    with pytest.raises(TypeError):
        rq(int(), methods, train_size, seed)

    with pytest.raises(TypeError):
        rq(dict(), methods, train_size, seed)

    with pytest.raises(TypeError):
        rq(set(), methods, train_size, seed)

    with pytest.raises(TypeError):
        rq(list(), methods, train_size, seed)

    with pytest.raises(TypeError):
        rq(tuple((0, 0)), methods, train_size, seed)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        rq('', methods, train_size, seed)

    # Test non-list value for method
    with pytest.raises(TypeError):
        rq(corpus, int(), train_size, seed)

    with pytest.raises(TypeError):
        rq(corpus, dict(), train_size, seed)

    with pytest.raises(TypeError):
        rq(corpus, set(), train_size, seed)

    with pytest.raises(TypeError):
        rq(corpus, tuple((0, 0)), train_size, seed)

    # Test non-int values for train_size
    with pytest.raises(TypeError):
        rq(corpus, methods, set(), seed)

    with pytest.raises(TypeError):
        rq(corpus, methods, dict(), seed)

    with pytest.raises(TypeError):
        rq(corpus, methods, list(), seed)

    with pytest.raises(TypeError):
        rq(corpus, methods, tuple((0, 0)), seed)

    with pytest.raises(TypeError):
        rq(corpus, methods, str(), seed)

    # Test non-positive values for train_size
    with pytest.raises(AttributeError):
        rq(corpus, methods, 0, seed)

    with pytest.raises(AttributeError):
        rq(corpus, methods, -1, seed)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        rq(corpus, methods, train_size, set())

    with pytest.raises(TypeError):
        rq(corpus, methods, train_size, list())

    with pytest.raises(TypeError):
        rq(corpus, methods, train_size, dict())

    with pytest.raises(TypeError):
        rq(corpus, methods, train_size, tuple((0, 0)))

    with pytest.raises(TypeError):
        rq(corpus, methods, train_size, str())

def test_retrieve_pickled_q():
    rpq = analyzer.retrieve_pickled_q

    q_filename = 'notacorpus_q.pickle'
    expected = [0 for _ in range(10)]
    with open(q_filename, 'wb') as f:
        pickle.dump(expected, f)

    # Make sure objects are equal
    q = rpq(q_filename)
    assert q == expected

    # Test non-string values
    with pytest.raises(TypeError):
        rpq(int())

    with pytest.raises(TypeError):
        rpq(dict())

    with pytest.raises(TypeError):
        rpq(set())

    with pytest.raises(TypeError):
        rpq(list())

    with pytest.raises(TypeError):
        rpq(tuple((0, 0)))

    # Test empty string
    with pytest.raises(AttributeError):
        rpq('')

    # Test nonexistant file
    os.remove(q_filename)
    assert not os.path.isfile(q_filename)
    with pytest.raises(AttributeError):
        rpq(q_filename)

def test_analyze_function():
    # Just need to make sure it runs, hard to test for specific outputs
    corpus = 'testamazon'
    methods = ['lc', 'r5']
    model = 'naive'
    train_size = 1000
    seed = 0

    try:
        # Test sklearn
        print(analyzer.analyze(corpus, methods, model, train_size, seed))

        # Test ankura
        model = 'ankura'
        print(analyzer.analyze(corpus, methods, model, train_size, seed))

    except:
        pytest.fail('Unexpected Error Occurred')

    finally:
        # Cleanup

        from preprocess import vocabulary
        vocab_filename = vocabulary.create_vocabulary_filename(corpus, methods, seed)
        imported_filename = importer.create_imported_corpus_filename(corpus, methods, train_size, seed)
        q_filename = analyzer.create_q_filename(corpus, methods, train_size, seed)
        ankura_results_filename = analyzer.generate_analyze_filename(corpus, methods, model, train_size, seed)
        sklearn_results_filename = analyzer.generate_analyze_filename(corpus, methods, 'naive', train_size, seed)

        home_dir = os.path.join(os.getenv('HOME'), '.preprocess')
        vocab_dir = os.path.join(home_dir, 'vocabulary')
        import_dir = os.path.join(home_dir, 'imported_corpora')
        q_dir = os.path.join(home_dir, 'ankura_q_objects')
        results_dir = analyzer.results_directory

        corpus_vocab_dir = os.path.join(vocab_dir, corpus)
        corpus_imported_dir = os.path.join(import_dir, corpus)
        corpus_q_dir = os.path.join(q_dir, corpus)
        corpus_results_dir = os.path.join(results_dir, corpus)

        os.remove(vocab_filename)
        os.remove(imported_filename)
        os.remove(q_filename)
        os.remove(ankura_results_filename)
        os.remove(sklearn_results_filename)

        os.rmdir(os.path.join(results_dir, corpus))

        for corpus in argmanager.valid_corpora:
            os.rmdir(os.path.join(vocab_dir, corpus))
            os.rmdir(os.path.join(import_dir, corpus))
            os.rmdir(os.path.join(q_dir, corpus))

        os.rmdir(import_dir)
        os.rmdir(vocab_dir)
        os.rmdir(q_dir)
        os.rmdir(results_dir)

        assert not os.path.isfile(vocab_filename)
        assert not os.path.isfile(imported_filename)
        assert not os.path.isfile(q_filename)
        assert not os.path.isfile(ankura_results_filename)
        assert not os.path.isfile(sklearn_results_filename)
        assert not os.path.isdir(os.path.join(results_dir, corpus))

        for corpus in argmanager.valid_corpora:
            assert not os.path.isdir(os.path.join(vocab_dir, corpus))
            assert not os.path.isdir(os.path.join(import_dir, corpus))
            assert not os.path.isdir(os.path.join(q_dir, corpus))

        assert not os.path.isdir(import_dir)
        assert not os.path.isdir(vocab_dir)
        assert not os.path.isdir(q_dir)
        assert not os.path.isdir(results_dir)

def test_run_ankura():
    # Just need to make sure it runs, hard to test for specific outputs

    # Need to test on real corpus because of anchor candidates
    corpus = 'testamazon'

    # Methods are meant to reduce vocabulary so it runs quickly
    methods = ['lc', 'r5']
    train_size = 4000
    seed = 0

    train, test = importer.import_corpus(corpus, methods, train_size, seed)
    Q = analyzer.retrieve_q(corpus, methods, train_size, seed)

    try:
        print(analyzer.run_ankura(train, test, Q))
    except:
        pytest.fail('Unexpected Error Occured')
    finally:

        # Cleanup
        from preprocess import vocabulary
        vocab_filename = vocabulary.create_vocabulary_filename(corpus, methods, seed)
        imported_filename = importer.create_imported_corpus_filename(corpus, methods, train_size, seed)
        q_filename = analyzer.create_q_filename(corpus, methods, train_size, seed)

        home_dir = os.path.join(os.getenv('HOME'), '.preprocess')
        vocab_dir = os.path.join(home_dir, 'vocabulary')
        import_dir = os.path.join(home_dir, 'imported_corpora')
        q_dir = os.path.join(home_dir, 'ankura_q_objects')

        corpus_vocab_dir = os.path.join(vocab_dir, corpus)
        corpus_imported_dir = os.path.join(import_dir, corpus)
        corpus_q_dir = os.path.join(q_dir, corpus)

        os.remove(vocab_filename)
        os.remove(imported_filename)
        os.remove(q_filename)

        for corpus in argmanager.valid_corpora:
            os.rmdir(os.path.join(vocab_dir, corpus))
            os.rmdir(os.path.join(import_dir, corpus))
            os.rmdir(os.path.join(q_dir, corpus))

        os.rmdir(import_dir)
        os.rmdir(vocab_dir)
        os.rmdir(q_dir)

        assert not os.path.isfile(vocab_filename)
        assert not os.path.isfile(imported_filename)
        assert not os.path.isfile(q_filename)

        for corpus in argmanager.valid_corpora:
            assert not os.path.isdir(os.path.join(vocab_dir, corpus))
            assert not os.path.isdir(os.path.join(import_dir, corpus))
            assert not os.path.isdir(os.path.join(q_dir, corpus))

        assert not os.path.isdir(import_dir)
        assert not os.path.isdir(vocab_dir)
        assert not os.path.isdir(q_dir)

def test_run_sklearn():
    # Just need to make sure it runs
    rs = analyzer.run_sklearn

    Corpus = importer.Corpus
    Document = importer.Document
    Token = importer.Token

    doc1 = Document([Token(0), Token(1), Token(2)], {'target' : 1})
    doc2 = Document([Token(1), Token(2), Token(3)], {'target' : 0})
    doc3 = Document([Token(2), Token(3), Token(4)], {'target' : 1})
    doc4 = Document([Token(2), Token(3), Token(4)], {'target' : 0})
    doc5 = Document([Token(2), Token(3), Token(4)], {'target' : 1})
    doc6 = Document([Token(2), Token(3), Token(4)], {'target' : 0})
    doc7 = Document([Token(2), Token(3), Token(4)], {'target' : 1})

    corpus = Corpus([doc1, doc2, doc3, doc4, doc5, doc6, doc7], {0, 1, 2, 3, 4}, dict())
    seed = 0

    try:
        # Naive bayes
        model = 'naive'
        rs(corpus, corpus, model)

        # SVM
        model = 'svm'
        rs(corpus, corpus, model)

        # K-NN
        model = 'knn'
        rs(corpus, corpus, model)

    except:
        pytest.fail('Unexpected error occurred')

    # Test non-string valies for model
    with pytest.raises(TypeError):
        rs(corpus, corpus, int())

    with pytest.raises(TypeError):
        rs(corpus, corpus, dict())

    with pytest.raises(TypeError):
        rs(corpus, corpus, list())

    with pytest.raises(TypeError):
        rs(corpus, corpus, set())

    with pytest.raises(TypeError):
        rs(corpus, corpus, tuple((0, 0)))

    # Test empty string for model
    with pytest.raises(AttributeError):
        rs(corpus, corpus, '')

    # Test none-types for train and test
    with pytest.raises(AttributeError):
        rs(None, corpus, model)

    with pytest.raises(AttributeError):
        rs(corpus, None, model)

    # Test empty iterators for train and test
    with pytest.raises(AttributeError):
        rs([], corpus, model)

    with pytest.raises(AttributeError):
        rs(corpus, [], model)

    with pytest.raises(AttributeError):
        rs(corpus, {}, model)

    with pytest.raises(AttributeError):
        rs({}, corpus, model)

def test_tfidf_conversion():
    ctt = analyzer.convert_to_tfidf

    Corpus = importer.Corpus
    Document = importer.Document
    Token = importer.Token

    doc1 = Document([Token(0), Token(1), Token(2)], {'target' : 0})
    doc2 = Document([Token(1), Token(2), Token(3)], {'target' : 0})
    doc3 = Document([Token(2), Token(3), Token(4)], {'target' : 0})

    corpus = Corpus([doc1, doc2, doc3], {0, 1, 2, 3, 4}, dict())
    vocab_size = len(corpus.vocabulary)

    expected = np.asarray([[0.4055, 0, -0.2877, 0, 0], [0, 0, -0.2877, 0, 0], [0, 0, -0.2877, 0, 0.4055]])
    tfidf = ctt(corpus, vocab_size)

    # Test for correct document frequency calculation
    import math

    for i in range(tfidf.shape[0]):
        for j in range(tfidf.shape[1]):
            assert math.isclose(expected[i, j], tfidf[i, j], rel_tol=0.01)

    doc1 = Document([Token(0), Token(0), Token(0)], {'target' : 0})
    doc2 = Document([Token(1), Token(1), Token(1)], {'target' : 0})
    doc3 = Document([Token(1), Token(2), Token(2)], {'target' : 0})

    corpus = Corpus([doc1, doc2, doc3], {0, 1, 2}, dict())
    vocab_size = len(corpus.vocabulary)

    expected = np.asarray([[1.2164, 0, 0], [0, 0, 0], [0, 0, 0.8109]])
    tfidf = ctt(corpus, vocab_size)

    # Test for correct term frequency calculation
    for i in range(tfidf.shape[0]):
        for j in range(tfidf.shape[1]):
            assert math.isclose(expected[i, j], tfidf[i, j], rel_tol=0.01)

    # Test for empty corpus
    corpus = Corpus([], {}, dict())
    assert not ctt(corpus, vocab_size)

    # Resetting corpus variable for parameter tests below
    corpus = Corpus([doc1, doc2, doc3], {0, 1, 2}, dict())

    # Test for non-int values for vocab_size
    with pytest.raises(TypeError):
        ctt(corpus, set())

    with pytest.raises(TypeError):
        ctt(corpus, dict())

    with pytest.raises(TypeError):
        ctt(corpus, list())

    with pytest.raises(TypeError):
        ctt(corpus, tuple((0, 0)))

    with pytest.raises(TypeError):
        ctt(corpus, str())

    # Test for zero value for vocab_size
    with pytest.raises(AttributeError):
        ctt(corpus, 0)

    # Test for negative vocab_size
    with pytest.raises(AttributeError):
        ctt(corpus, -1)

    # Test for invalid corpus
    badcorpus = namedtuple('Badcorpus', 'notdocuments vocabulary metadata')
    b = badcorpus([doc1, doc2, doc3], {0, 1, 2}, dict())
    with pytest.raises(AttributeError):
        ctt(b, vocab_size)

    baddocs = namedtuple('baddocs', 'nottokens metadata')
    doc1 = baddocs([Token(0), Token(0), Token(0)], {'target' : 0})
    doc2 = baddocs([Token(1), Token(1), Token(1)], {'target' : 0})
    doc3 = baddocs([Token(1), Token(2), Token(2)], {'target' : 0})

    corpus = Corpus([doc1, doc2, doc3], {0, 1, 2}, dict())

    with pytest.raises(AttributeError):
        ctt(corpus, vocab_size)
