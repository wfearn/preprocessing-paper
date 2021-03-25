from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, lil_matrix, dok_matrix, vstack
import numpy as np
from collections import namedtuple, defaultdict
import pickle
import os
import time
import scipy
import sklearn
import ankura
import preprocess
from preprocess import importer

try:
    from vowpalwabbit.sklearn_vw import VWClassifier
except ImportError:
    VWClassifier = None

preprocess_dir = os.path.join(os.getenv('HOME'), '.preprocess')
q_directory = os.path.join(preprocess_dir, 'ankura_q_objects')
results_directory = os.path.join(preprocess_dir, 'results')

THETA_ATTR = 'theta'
Results = namedtuple('Results', 'vocabsize tokens accuracy precision recall traintime testtime topictime anchortime seed')

scoring_dict = {
                    'accuracy' : make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall'   : make_scorer(recall_score),
               }

def generate_analyze_filename(corpus, methods, model, train_size, seed):
    if type(corpus) is not str:
        raise TypeError('Non-string value invalid for parameter \'corpus\'')
    if not corpus:
        raise AttributeError('Parameter \'corpus\' must be non-empty')
    if type(model) is not str:
        raise TypeError('Non-string value invalid for parameter \'model\'')
    if not model:
        raise AttributeError('Parameter \'model\' must be non-empty')
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

    from preprocess import argmanager as args
    corpus_dir = os.path.join(results_directory, corpus)

    full_corpus_name = args.processed_corpus_name(corpus, methods)
    base_filename = f'{full_corpus_name}_{model}_corpus{train_size}_results_{seed}.pickle'

    return os.path.join(corpus_dir, base_filename)

def analyze(corpus, methods, model, train_size, seed):
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

    np.random.seed(seed)

    train, test = importer.import_corpus(corpus, methods, train_size, seed)

    if model == 'ankura':
        Q = retrieve_q(corpus, methods, train_size, seed)
        accuracy, precision, recall, train_time, test_time, topic_time, anchor_time = run_ankura(train, test, Q)

    else:
        accuracy, precision, recall, train_time, test_time, topic_time, anchor_time = run_sklearn(train, test, model)


    results = Results(len(train.vocabulary), train.metadata['total_tokens'], accuracy, precision, recall, train_time, test_time, topic_time, anchor_time, seed)

    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    corpus_dir = os.path.join(results_directory, corpus)
    if not os.path.isdir(corpus_dir):
        os.mkdir(corpus_dir)

    results_filename = generate_analyze_filename(corpus, methods, model, train_size, seed)
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    return results

def get_targets(corpus):
    # Corpus is namedtuple, type checking is done implicitly

    # Get rid of this hack asap, only using it so I don't have to re-import corpora
    all_targets = set([d.metadata['target'] for d in corpus.documents])
    if len(all_targets) == 5: #This means we're dealing with amazon
        targets = [1 if d.metadata['target'] > 4 else 0 for d in corpus.documents]
    else:
        targets = [d.metadata['target'] for d in corpus.documents]

    return targets

# Implemented tfidf function to allow for more control over vocabulary
# and what counts as a word

def convert_to_tfidf(corpus, vocab_size):
    if type(vocab_size) is not int:
        raise TypeError('Non-int value not valid for parameter \'vocab_size\'')
    if vocab_size <= 0:
        raise AttributeError('Parameter \'vocab_size\' must be positive')

    ## Rows are features (vocabulary in this case)
    data_matrix = defaultdict(np.float32)

    token_indices = dict()
    N = len(corpus.documents)

    for i, doc in enumerate(corpus.documents):

        # Makes it easier to predict output

        tokens = {t.token for t in doc.tokens}
        for token in tokens:
            if token not in token_indices.keys():
                # Token indices could be larger than vocab size because they come from
                # A larger corpus
                token_indices[token] = [len(token_indices), 0]

            # Keep track of doc frequency
            token_indices[token][1] += 1

        for token in doc.tokens:
            t_i = token_indices[token.token][0]
            # Term frequency
            data_matrix[(i, t_i)] += 1

    for _, token_data in token_indices.items():
        t_i = token_data[0]
        df = token_data[1]
        idf = np.log(N / (1 + df))

        for i in range(len(corpus.documents)):
            key = (i, t_i)
            if key not in data_matrix.keys(): continue
            data_matrix[key] *= idf

    # using a dok lets us easily convert to a csr, which saves on space
    data_dok = dok_matrix((len(corpus.documents), vocab_size), dtype=np.float32)
    # regular update method got changed to not allow assignment of multiple values at once
    data_dok._update(data_matrix)

    return normalize(data_dok.tocsr())

# I don't think I need a seed here.
def run_sklearn(train, test, model):
    if type(model) is not str:
        raise TypeError('Non-str value invalid for parameter \'model\'')
    if not model:
        raise AttributeError('Empty string invalid for parameter \'model\'')
    if not train:
        raise AttributeError('None-type or empty object invalid for parameter \'train\'')
    if not test:
        raise AttributeError('None-type or empty object invalid for parameter \'test\'')

    sklearn_model_dict = {
                            'naive' : BernoulliNB(),
                            'svm'   : LinearSVC(),
                            'knn'   : KNeighborsClassifier(),
                         }

    vocab_size = len(train.vocabulary)

    raw_train = [' '.join([str(t.token) for t in doc.tokens]) for doc in train.documents]
    raw_test = [' '.join([str(t.token) for t in doc.tokens]) for doc in test.documents]

    voc = dict()
    for doc in raw_train:
        for t in doc.split():
            if t not in voc.keys():
                voc[t] = len(voc)

    tfv = TfidfVectorizer(vocabulary=voc)
    train_matrix = tfv.fit_transform(raw_train)
    test_matrix = tfv.transform(raw_test)

    # Hacky, should fix this
    X = vstack([train_matrix, test_matrix])

    train_targets = get_targets(train)
    test_targets = get_targets(test)

    if model == 'vowpal':
        assert VWClassifier is not None
        sklearn_model_dict[model] = VWClassifier()

        # Vowpal Wabbit requires this format
        train_targets = [-1 if t == 0 else 1 for t in train_targets]
        test_targets = [-1 if t == 0 else 1 for t in test_targets]


    train_targets.extend(test_targets)
    Y = train_targets

    sklearn_model = sklearn_model_dict[model]

    # random state 0 gives us consistency
    rs = ShuffleSplit(n_splits=5, test_size=1000, random_state=0)
    results = cross_validate(sklearn_model, X, Y, cv=rs, scoring=scoring_dict, error_score=0.0)

    accuracy = np.mean(results['test_accuracy'])
    precision = np.mean(results['test_precision'])
    recall = np.mean(results['test_recall'])

    train_time = np.mean(results['fit_time'])
    test_time = np.mean(results['score_time'])

    # We need to compare to ankura
    topic_time = 0
    anchor_time = 0

    return accuracy, precision, recall, train_time, test_time, topic_time, anchor_time

# Hard to test this because I'd need actual topics.
def convert_to_data_matrix(corpus, topics):
    ankura.topic.gensim_assign(corpus, topics, theta_attr=THETA_ATTR)

    # Topics should be K x V
    matrix = np.zeros((len(corpus.documents), topics.shape[1]))

    for i, doc in enumerate(corpus.documents):
        matrix[i, :] = np.log(doc.metadata[THETA_ATTR] + 1e-30)

    return matrix

def run_ankura(train, test, Q, K=80):

    anchor_start = time.time()
    anchors = ankura.anchor.gram_schmidt_anchors(train, Q, K)
    anchor_time = time.time() - anchor_start

    topic_start = time.time()
    topics = ankura.anchor.recover_topics(Q, anchors)
    topic_time = time.time() - topic_start

    train_matrix = convert_to_data_matrix(train, topics)
    test_matrix = convert_to_data_matrix(test, topics)

    X = vstack([train_matrix, test_matrix])

    train_targets = get_targets(train)
    test_targets = get_targets(test)

    train_targets.extend(test_targets)
    Y = train_targets

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()

    rs = ShuffleSplit(n_splits=5, test_size=1000, random_state=0)
    results = cross_validate(lr, X, Y, cv=rs, scoring=scoring_dict, error_score=0.0)

    accuracy = np.mean(results['test_accuracy'])
    precision = np.mean(results['test_precision'])
    recall = np.mean(results['test_recall'])

    train_time = np.mean(results['fit_time'])
    test_time = np.mean(results['score_time'])

    return accuracy, precision, recall, train_time, test_time, topic_time, anchor_time

def retrieve_pickled_q(q_filename):
    if type(q_filename) is not str:
        raise TypeError('Non-string value invalid for parameter \'q_filename\'')
    if not q_filename:
        raise AttributeError('Parameter \'q_filename\' must be non-empty')
    if not os.path.isfile(q_filename):
        raise AttributeError('Parameter \'q_filename\' must exist')

    with open(q_filename, 'rb') as f:
        q = pickle.load(f)
    return q

def create_q_directories():
    if not os.path.isdir(q_directory):
        os.mkdir(q_directory)

    from preprocess import argmanager as arg

    for c in arg.valid_corpora:
        c_dir = os.path.join(q_directory, c)

        if not os.path.isdir(c_dir):
            os.mkdir(c_dir)

def retrieve_q(corpus, methods, train_size, seed):
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

    from ankura import anchor
    from preprocess import importer

    # Create Q dirs
    if not os.path.isdir(q_directory):
        create_q_directories()

    q_filename = create_q_filename(corpus, methods, train_size, seed)

    if os.path.isfile(q_filename):
        return retrieve_pickled_q(q_filename)

    # Corpus should already be imported
    train, test = importer.import_corpus(corpus, methods, train_size, seed)

    if len(train.vocabulary) > 100000:
        raise MemoryError('Vocabulary size greater than 100000 are too large for building q')

    Q = anchor.build_supervised_cooccurrence(train, 'target', set(range(len(train.documents))))

    with open(q_filename, 'wb') as f:
        pickle.dump(Q, f, protocol=4)

    return Q

def create_q_filename(corpus, methods, train_size, seed):
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

    base_filename = f'{corpus}/{arg.processed_corpus_name(corpus, methods)}_s{seed}_corpus{train_size}_Q.pickle'
    return os.path.join(q_directory, base_filename)
