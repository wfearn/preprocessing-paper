from preprocess import importer
from preprocess import vocabulary as v
import gzip
import os
import pytest

home_dir = os.path.join(os.getenv('HOME'), '.preprocess')
import_corpus_dir = os.path.join(home_dir, 'imported_corpora')

def test_filter_dict():
    fd = importer.filter_dict

    # Test default behavior for default corpora
    target_filter = fd['amazon']
    values = [1, 2, 3, 4, 'cat', 'dog', 'francis']

    for val in values:
        # Should be identity function if not reddit
        assert target_filter(val) == val

    target_filter = fd['testamazon']
    for val in values:
        assert target_filter(val) == val

    target_filter = fd['apnews']
    for val in values:
        assert target_filter(val) == val

    target_filter = fd['testapnews']
    for val in values:
        assert target_filter(val) == val

    target_filter = fd['twitter']
    for val in values:
        assert target_filter(val) == val

    target_filter = fd['testtwitter']
    for val in values:
        assert target_filter(val) == val

    # Test proper behavior for reddit
    target_filter = fd['reddit']
    assert not target_filter(-5)
    assert not target_filter(-3)
    assert not target_filter(-1)
    assert not target_filter(0)
    assert not target_filter(1)

    assert target_filter(2)
    assert target_filter(3)
    assert target_filter(4)
    assert target_filter(1000)

def test_filename_creation():
    cicf = importer.create_imported_corpus_filename

    corpus = 'amazon'
    methods = ['lc']
    train_size = 5000
    seed = 0

    import_amazon_dir = os.path.join(import_corpus_dir, 'amazon')
    import_reddit_dir = os.path.join(import_corpus_dir, 'reddit')

    # Test base functionality
    expected = 'amazon_lower_s0_corpus5000.pickle'
    assert cicf(corpus, methods, train_size, seed) == os.path.join(import_amazon_dir, expected)

    # Test changing train_size
    train_size = 673894
    expected = 'amazon_lower_s0_corpus673894.pickle'
    assert cicf(corpus, methods, train_size, seed) == os.path.join(import_amazon_dir, expected)

    # Test changing methods
    methods = ['np']
    expected = 'amazon_nopunct_s0_corpus673894.pickle'
    assert cicf(corpus, methods, train_size, seed) == os.path.join(import_amazon_dir, expected)

    # Test changing seed
    seed = 6
    expected = 'amazon_nopunct_s6_corpus673894.pickle'
    assert cicf(corpus, methods, train_size, seed) == os.path.join(import_amazon_dir, expected)

    # Test changing corpus
    corpus = 'reddit'
    expected = 'reddit_nopunct_s6_corpus673894.pickle'
    assert cicf(corpus, methods, train_size, seed) == os.path.join(import_reddit_dir, expected)

    # Test no methods
    expected = 'reddit_s6_corpus673894.pickle'
    assert cicf(corpus, [], train_size, seed) == os.path.join(import_reddit_dir, expected)

    # Test non-string for corpus
    with pytest.raises(TypeError):
        cicf(int(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cicf(dict(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cicf(set(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cicf(list(), methods, train_size, seed)

    with pytest.raises(TypeError):
        cicf(tuple((0, 0)), methods, train_size, seed)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        cicf('', methods, train_size, seed)

    # Test non-list value for method
    with pytest.raises(TypeError):
        cicf(corpus, int(), train_size, seed)

    with pytest.raises(TypeError):
        cicf(corpus, dict(), train_size, seed)

    with pytest.raises(TypeError):
        cicf(corpus, set(), train_size, seed)

    with pytest.raises(TypeError):
        cicf(corpus, tuple((0, 0)), train_size, seed)

    # Test non-int values for train_size
    with pytest.raises(TypeError):
        cicf(corpus, methods, set(), seed)

    with pytest.raises(TypeError):
        cicf(corpus, methods, dict(), seed)

    with pytest.raises(TypeError):
        cicf(corpus, methods, list(), seed)

    with pytest.raises(TypeError):
        cicf(corpus, methods, tuple((0, 0)), seed)

    with pytest.raises(TypeError):
        cicf(corpus, methods, str(), seed)

    # Test non-positive values for train_size
    with pytest.raises(AttributeError):
        cicf(corpus, methods, 0, seed)

    with pytest.raises(AttributeError):
        cicf(corpus, methods, -1, seed)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        cicf(corpus, methods, train_size, set())

    with pytest.raises(TypeError):
        cicf(corpus, methods, train_size, list())

    with pytest.raises(TypeError):
        cicf(corpus, methods, train_size, dict())

    with pytest.raises(TypeError):
        cicf(corpus, methods, train_size, tuple((0, 0)))

    with pytest.raises(TypeError):
        cicf(corpus, methods, train_size, str())

def test_is_corpus_testable():
    ict = importer.is_corpus_testable

    # Test base functionality
    assert ict('reddit')

    # Test other testable corpus
    assert ict('amazon')

    # Test non-testable corpus
    assert not ict('notacorpus')

    # Test nonsensical corpus
    assert not ict('alidjsfls')

    # Test non-string values
    with pytest.raises(TypeError):
        ict(int())

    with pytest.raises(TypeError):
        ict(dict())

    with pytest.raises(TypeError):
        ict(set())

    with pytest.raises(TypeError):
        ict(list())

    with pytest.raises(TypeError):
        ict(tuple((0, 0)))

    # Test empty string
    with pytest.raises(AttributeError):
        ict('')

def test_import_corpus_function():
    import pickle

    ic = importer.import_corpus

    # Corpus Setup
    notacorpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/notacorpus')
    notacorpus_filepath = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')

    assert not os.path.isdir(notacorpus_dir)
    assert not os.path.isfile(notacorpus_filepath)

    s = "word1 word2 word3 word4 word5"
    os.mkdir(notacorpus_dir)
    with gzip.open(notacorpus_filepath, 'wb') as f:
        f.write(f'{s}\n{s}'.encode('utf-8'))

    # Vocab Setup
    vocab_dir = os.path.join(os.getenv('HOME'), '.preprocess/vocabulary')
    notacorpus_vocab_dir = os.path.join(vocab_dir, 'notacorpus')
    notacorpus_vocab_fp = os.path.join(notacorpus_vocab_dir, 'notacorpus_vocabulary.pickle')

    assert not os.path.isdir(notacorpus_vocab_dir)
    assert not os.path.isfile(notacorpus_vocab_fp)

    vocab_dict = {word : [int(word[-1]), 2, int(word[-1])] for word in s.split()}
    vocab_info = v.VocabInfo()
    vocab_info.dictionary = vocab_dict

    os.mkdir(vocab_dir)
    os.mkdir(notacorpus_vocab_dir)
    with open(notacorpus_vocab_fp, 'wb') as f:
        pickle.dump(vocab_info, f)

    from preprocess import argmanager as arg
    import_corpus_dir = importer.import_corpus_dir
    assert not os.path.isdir(import_corpus_dir)
    for c in arg.valid_corpora:
        assert not os.path.isdir(os.path.join(import_corpus_dir, c))

    corpus = 'notacorpus'
    methods = []
    train_size = 1
    seed = 0
    test_size = 1

    imported_corpus_filename = importer.create_imported_corpus_filename(corpus, methods, train_size, seed)
    assert not os.path.isfile(imported_corpus_filename)

    # Test with no filtering
    train, test = ic(corpus, methods, train_size, seed, test_size=test_size)

    # Reducing all tokens by 1 because of how they will be re-assigned
    # token values
    expected_train = [[0, 1, 2, 3, 4]]
    expected_test = [[0, 1, 2, 3, 4]]
    expected_vocabulary = {1, 2, 3, 4, 5}
    expected_metadata = { 'total_tokens' : 5 }

    # Test correct pickle creation
    assert os.path.isfile(imported_corpus_filename)

    # Test correct directory creation
    assert os.path.isdir(import_corpus_dir)
    for c in arg.valid_corpora:
        assert os.path.isdir(os.path.join(import_corpus_dir, c))

    # Test correct train set
    assert train.vocabulary == expected_vocabulary
    assert train.metadata == expected_metadata
    for i, doc in enumerate(train.documents):
        assert not doc.metadata['target']
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_train[i][j]

    # Test correct test set
    assert not test.vocabulary
    assert not test.metadata
    for i, doc in enumerate(test.documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_test[i][j]

    os.remove(imported_corpus_filename)
    assert not os.path.isfile(imported_corpus_filename)

    # Test with some filtering
    os.remove(notacorpus_filepath)
    assert not os.path.isfile(notacorpus_filepath)

    # Setup corpus
    notacorpus_filepath = os.path.join(notacorpus_dir, 'notacorpus_r3.txt.gz')
    with gzip.open(notacorpus_filepath, 'wb') as f:
        f.write(f'{s}\n{s}'.encode('utf-8'))

    # Setup vocab
    os.remove(notacorpus_vocab_fp)
    assert not os.path.isfile(notacorpus_vocab_fp)
    notacorpus_vocab_fp = os.path.join(notacorpus_vocab_dir, 'notacorpus_r3_vocabulary.pickle')
    with open(notacorpus_vocab_fp, 'wb') as f:
        pickle.dump(vocab_info, f)

    methods = ['r3']
    train, test = ic(corpus, methods, train_size, seed, test_size=test_size)

    imported_corpus_filename = importer.create_imported_corpus_filename(corpus, methods, train_size, seed)

    expected_train = [[0, 1, 2]]
    expected_test = [[0, 1, 2]]
    expected_vocabulary = {3, 4, 5}
    expected_metadata = {'total_tokens' : 3}

    assert os.path.isfile(imported_corpus_filename)
    os.remove(imported_corpus_filename)
    assert not os.path.isfile(imported_corpus_filename)

    # Test correct train set
    assert train.vocabulary == expected_vocabulary
    assert train.metadata == expected_metadata
    for i, doc in enumerate(train.documents):
        assert not doc.metadata['target']
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_train[i][j]

    # Test correct test set
    assert not test.vocabulary
    assert not test.metadata
    for i, doc in enumerate(test.documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_test[i][j]

    # Test to ensure that words unique to test set get filtered
    tr = "word0 word1 word2 word3 word4 word5"

    # word6 and word7 should get filtered because of how we constructed the vocabulary above
    te = "word3 word4 word5 word6 word7 word8"
    os.remove(notacorpus_filepath)
    assert not os.path.isfile(notacorpus_filepath)

    # Setup corpus
    notacorpus_filepath = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')
    with gzip.open(notacorpus_filepath, 'wb') as f:
        f.write(f'{tr}\n{te}'.encode('utf-8'))

    # Setup vocab
    vocab_dict = {word : [int(word[-1]), 2, int(word[-1])] for word in tr.split()}
    vocab_info = v.VocabInfo()
    vocab_info.dictionary = vocab_dict

    os.remove(notacorpus_vocab_fp)
    assert not os.path.isfile(notacorpus_vocab_fp)
    notacorpus_vocab_fp = os.path.join(notacorpus_vocab_dir, 'notacorpus_vocabulary.pickle')
    with open(notacorpus_vocab_fp, 'wb') as f:
        pickle.dump(vocab_info, f)

    methods = []

    # Change the seed so documents when shuffled maintain the same order
    seed = 1
    train, test = ic(corpus, methods, train_size, seed, test_size=test_size)

    imported_corpus_filename = importer.create_imported_corpus_filename(corpus, methods, train_size, seed)
    expected_train = [[0, 1, 2, 3, 4]]
    expected_test = [[2, 3, 4]]

    # We expect 0 to be filtered because it has a frequency of 0
    expected_vocabulary = {1, 2, 3, 4, 5}
    expected_metadata = {'total_tokens' : 5}

    assert os.path.isfile(imported_corpus_filename)
    os.remove(imported_corpus_filename)
    assert not os.path.isfile(imported_corpus_filename)

    # Test correct train set
    assert train.vocabulary == expected_vocabulary
    assert train.metadata == expected_metadata
    for i, doc in enumerate(train.documents):
        assert not doc.metadata['target']
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_train[i][j]

    # Test correct test set
    assert not test.vocabulary
    assert not test.metadata
    for i, doc in enumerate(test.documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_test[i][j]

    # Clean up
    for c in arg.valid_corpora:
        os.rmdir(os.path.join(import_corpus_dir, c))
        assert not os.path.isdir(os.path.join(import_corpus_dir, c))
    os.rmdir(import_corpus_dir)
    assert not os.path.isdir(import_corpus_dir)

    os.remove(notacorpus_filepath)
    os.rmdir(notacorpus_dir)
    assert not os.path.isfile(notacorpus_filepath)
    assert not os.path.isdir(notacorpus_dir)

    os.remove(notacorpus_vocab_fp)
    os.rmdir(notacorpus_vocab_dir)
    assert not os.path.isfile(notacorpus_vocab_fp)
    assert not os.path.isdir(notacorpus_vocab_dir)

    os.rmdir(vocab_dir)
    assert not os.path.isdir(vocab_dir)

def test_retrieve_imported_corpus():
    import pickle
    ric = importer.retrieve_imported_corpus

    imported_corpus_dir = importer.import_corpus_dir
    notacorpus_corpus_dir = os.path.join(import_corpus_dir, 'notacorpus')
    corpus_data = ['doc1', 'doc2', 'doc3', 'doc4']
    corpus = 'notacorpus'
    methods = []
    train_size = 500
    seed = 0
    notacorpus_filename = os.path.join(notacorpus_corpus_dir, f'{corpus}_s{seed}_corpus{train_size}.pickle')

    os.mkdir(imported_corpus_dir)
    os.mkdir(notacorpus_corpus_dir)
    with open(notacorpus_filename, 'wb') as f:
        pickle.dump(corpus_data, f)

    train = ric(corpus, methods, train_size, seed)

    assert train == corpus_data

    os.remove(notacorpus_filename)
    os.rmdir(notacorpus_corpus_dir)
    os.rmdir(imported_corpus_dir)
    assert not os.path.isfile(notacorpus_filename)
    assert not os.path.isdir(notacorpus_corpus_dir)
    assert not os.path.isdir(imported_corpus_dir)

    # Test non-string for corpus
    with pytest.raises(TypeError):
        ric(int(), methods, train_size, seed)

    with pytest.raises(TypeError):
        ric(dict(), methods, train_size, seed)

    with pytest.raises(TypeError):
        ric(set(), methods, train_size, seed)

    with pytest.raises(TypeError):
        ric(list(), methods, train_size, seed)

    with pytest.raises(TypeError):
        ric(tuple((0, 0)), methods, train_size, seed)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        ric('', methods, train_size, seed)

    # Test non-list value for method
    with pytest.raises(TypeError):
        ric(corpus, int(), train_size, seed)

    with pytest.raises(TypeError):
        ric(corpus, dict(), train_size, seed)

    with pytest.raises(TypeError):
        ric(corpus, set(), train_size, seed)

    with pytest.raises(TypeError):
        ric(corpus, tuple((0, 0)), train_size, seed)

    # Test non-int values for train_size
    with pytest.raises(TypeError):
        ric(corpus, methods, set(), seed)

    with pytest.raises(TypeError):
        ric(corpus, methods, dict(), seed)

    with pytest.raises(TypeError):
        ric(corpus, methods, list(), seed)

    with pytest.raises(TypeError):
        ric(corpus, methods, tuple((0, 0)), seed)

    with pytest.raises(TypeError):
        ric(corpus, methods, str(), seed)

    # Test non-positive values for train_size
    with pytest.raises(AttributeError):
        ric(corpus, methods, 0, seed)

    with pytest.raises(AttributeError):
        ric(corpus, methods, -1, seed)

    # Test non-int values for seed
    with pytest.raises(TypeError):
        ric(corpus, methods, train_size, set())

    with pytest.raises(TypeError):
        ric(corpus, methods, train_size, list())

    with pytest.raises(TypeError):
        ric(corpus, methods, train_size, dict())

    with pytest.raises(TypeError):
        ric(corpus, methods, train_size, tuple((0, 0)))

    with pytest.raises(TypeError):
        ric(corpus, methods, train_size, str())

def test_import_corpus_pickle_retrieval():
    import pickle

    ic = importer.import_corpus

    imported_corpus_dir = importer.import_corpus_dir
    notacorpus_corpus_dir = os.path.join(import_corpus_dir, 'notacorpus')
    corpus_data = ['doc1', 'doc2', 'doc3', 'doc4']
    corpus = 'notacorpus'
    methods = []
    train_size = 500
    seed = 0
    notacorpus_filename = os.path.join(notacorpus_corpus_dir, f'{corpus}_s{seed}_corpus{train_size}.pickle')

    os.mkdir(imported_corpus_dir)
    os.mkdir(notacorpus_corpus_dir)
    with open(notacorpus_filename, 'wb') as f:
        pickle.dump((corpus_data, 0), f)

    train, test = ic(corpus, methods, train_size, seed)

    assert train == corpus_data
    assert not test

    os.remove(notacorpus_filename)
    os.rmdir(notacorpus_corpus_dir)
    os.rmdir(imported_corpus_dir)
    assert not os.path.isfile(notacorpus_filename)
    assert not os.path.isdir(notacorpus_corpus_dir)
    assert not os.path.isdir(imported_corpus_dir)

def test_extract_documents():
    ed = importer.extract_documents

    subset = ['word1 word2 word3', 'word4 word5 word6']
    testable = False

    vinfo = v.VocabInfo()
    vinfo.dictionary = {word : [int(word[-1]), 1, 1] for doc in subset for word in doc.split()}
    reject = lambda x : x.endswith('6')
    corpus = 'testamazon'

    # This tests the reject function
    expected_documents = [[0, 1, 2], [3, 4]]
    expected_total_tokens = 5
    # We keep the original token numbers but assign them different numbers
    # In the documents
    expected_small_vocab = {1:0, 2:1, 3:2, 4:3, 5:4}

    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    possible_targets = {0, 1}

    # Test correct documents and targets
    for i, doc in enumerate(documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_documents[i][j]
        assert doc.metadata['target'] in possible_targets

    # Test correct number of total tokens
    assert total_tokens == expected_total_tokens

    # Test correct small vocabulary
    assert expected_small_vocab == small_vocab

    s = "{\"reviewText\": \"word1 word2 word3\", \"overall\": 5.0}"
    subset = [s, s]
    testable = True

    # Always returns false
    reject = lambda x : type(total_tokens) is not int

    expected_documents = [[0, 1, 2], [0, 1, 2]]
    expected_total_tokens = 6
    expected_small_vocab = {1:0, 2:1, 3:2}
    expected_target = [5, 5]

    # Test when testable is true
    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    for i, doc in enumerate(documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_documents[i][j]
        assert doc.metadata['target'] == expected_target[i]

    # Test rejecting everything
    s = "{\"reviewText\": \"word1 word2 word3\", \"overall\": 5.0}"
    subset = [s, s]
    testable = True

    # Always returns true
    reject = lambda x : type(total_tokens) is int

    expected_documents = [[0, 1, 2], [0, 1, 2]]
    expected_total_tokens = 6
    expected_small_vocab = {1:0, 2:1, 3:2}
    expected_target = [5, 5]

    # Test when testable is true
    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    assert not documents
    assert not total_tokens
    assert not small_vocab

    # Test with Rare word filtering
    s = "{\"reviewText\": \"the and of\", \"overall\": 5.0}"
    subset = [s, s]
    testable = True

    stop = v.generate_stopword_list()
    reject = lambda x : x in stop

    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    assert not documents
    assert not total_tokens
    assert not small_vocab

    # Make sure it works for reddit
    corpus = 'reddit'
    s = "{\"body\": \"word1 word2 word3\", \"score\": 5.0}"
    subset = [s, s]
    testable = True

    reject = lambda x : type(total_tokens) is not int

    expected_documents = [[0, 1, 2], [0, 1, 2]]
    expected_total_tokens = 6
    expected_small_vocab = {1:0, 2:1, 3:2}
    expected_target = [1, 1]

    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    for i, doc in enumerate(documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_documents[i][j]
        assert doc.metadata['target'] == expected_target[i]

    # Test with reddit and score 1 to ensure correct output
    s = "{\"body\": \"word1 word2 word3\", \"score\": 1.0}"
    subset = [s, s]
    testable = True

    reject = lambda x : type(total_tokens) is not int

    expected_documents = [[0, 1, 2], [0, 1, 2]]
    expected_total_tokens = 6
    expected_small_vocab = {1:0, 2:1, 3:2}
    expected_target = [0, 0]

    documents, total_tokens, small_vocab = ed(subset, testable, vinfo, reject, corpus)
    for i, doc in enumerate(documents):
        for j, token in enumerate(doc.tokens):
            assert token.token == expected_documents[i][j]
        assert doc.metadata['target'] == expected_target[i]

    # Test correct number of total tokens
    assert expected_total_tokens == total_tokens

    # Test correct small vocabulary
    assert small_vocab == expected_small_vocab

    # Test non-lists for subset parameter
    with pytest.raises(TypeError):
        ed(int(), testable, vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(dict(), testable, vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(set(), testable, vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(tuple((0, 0)), testable, vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(str(), testable, vinfo, reject, corpus)

    # Test empty list for subset
    with pytest.raises(AttributeError):
        ed([], testable, vinfo, reject, corpus)

    # Test non-booleans for parameter testable
    with pytest.raises(TypeError):
        ed(subset, int(), vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, dict(), vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, set(), vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, tuple((0, 0)), vinfo, reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, str(), vinfo, reject, corpus)

    # Test non-VocabInfo for subset vocabulary
    with pytest.raises(TypeError):
        ed(subset, testable, int(), reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, testable, dict(), reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, testable, set(), reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, testable, tuple((0, 0)), reject, corpus)

    with pytest.raises(TypeError):
        ed(subset, testable, str(), reject, corpus)

    # Test non-string values for corpus
    with pytest.raises(TypeError):
        ed(subset, testable, vinfo, reject, int())

    with pytest.raises(TypeError):
        ed(subset, testable, vinfo, reject, dict())

    with pytest.raises(TypeError):
        ed(subset, testable, vinfo, reject, set())

    with pytest.raises(TypeError):
        ed(subset, testable, vinfo, reject, tuple((0, 0)))

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        ed(subset, testable, vinfo, reject, '')

def test_create_imported_corpus_dirs():
    from preprocess import argmanager as arg

    cicd = importer.create_imported_corpus_dirs

    # Test basic functionality
    import_corpus_dir = importer.import_corpus_dir
    assert not os.path.isdir(import_corpus_dir)
    for c in arg.valid_corpora:
        assert not os.path.isdir(os.path.join(import_corpus_dir, c))

    cicd()

    assert os.path.isdir(import_corpus_dir)
    for c in arg.valid_corpora:
        assert os.path.isdir(os.path.join(import_corpus_dir, c))

    for c in arg.valid_corpora:
        os.rmdir(os.path.join(import_corpus_dir, c))
    os.rmdir(import_corpus_dir)

    for c in arg.valid_corpora:
        assert not os.path.isdir(os.path.join(import_corpus_dir, c))
    assert not os.path.isdir(import_corpus_dir)

def test_retrieve_preprocessed_corpus():
    rpc = importer.retrieve_preprocessed_corpus
    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    notacorpus_dir = os.path.join(os.getenv('HOME'), '.preprocess/corpora/notacorpus')
    notacorpus_filepath = os.path.join(notacorpus_dir, 'notacorpus.txt.gz')

    assert not os.path.isdir(notacorpus_dir)
    assert not os.path.isfile(notacorpus_filepath)

    os.mkdir(notacorpus_dir)
    with gzip.open(notacorpus_filepath, 'wb') as f:
        f.write(f'{s}\n{s}'.encode('utf-8'))

    corpus = 'notacorpus'
    methods = []

    documents = rpc(corpus, methods)

    # Test metadata
    assert len(documents) == 2

    # Test equivalence
    assert documents == [s, s]

    # Test non-string for corpus
    with pytest.raises(TypeError):
        rpc(int(), methods)

    with pytest.raises(TypeError):
        rpc(dict(), methods)

    with pytest.raises(TypeError):
        rpc(set(), methods)

    with pytest.raises(TypeError):
        rpc(list(), methods)

    with pytest.raises(TypeError):
        rpc(tuple((0, 0)))

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        rpc('', methods)

    # Test non-list value for method
    with pytest.raises(TypeError):
        rpc(corpus, int())

    with pytest.raises(TypeError):
        rpc(corpus, dict())

    with pytest.raises(TypeError):
        rpc(corpus, set())

    with pytest.raises(TypeError):
        rpc(corpus, tuple((0, 0)))

    os.remove(notacorpus_filepath)
    os.rmdir(notacorpus_dir)

    assert not os.path.isfile(notacorpus_filepath)
    assert not os.path.isdir(notacorpus_dir)

def test_get_target():
    gt = importer.get_target

    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    # Test normal functionality
    assert gt(s) == 5

    # Test different score
    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 4.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"
    assert gt(s) == 4

    # Test non-string values for document
    with pytest.raises(TypeError):
        gt(int())

    with pytest.raises(TypeError):
        gt(dict())

    with pytest.raises(TypeError):
        gt(set())

    with pytest.raises(TypeError):
        gt(list())

    with pytest.raises(TypeError):
        gt(tuple((0, 0)))

    # Test empty string for document
    with pytest.raises(AttributeError):
        gt('')

    # Test non-string values for tag
    with pytest.raises(TypeError):
        gt(s, int())

    with pytest.raises(TypeError):
        gt(s, dict())

    with pytest.raises(TypeError):
        gt(s, set())

    with pytest.raises(TypeError):
        gt(s, list())

    with pytest.raises(TypeError):
        gt(s, tuple((0, 0)))

    # Test empty string for tag
    with pytest.raises(AttributeError):
        gt(s, '')
