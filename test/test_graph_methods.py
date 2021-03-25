import pytest
import os
from preprocess import graph
from preprocess import argmanager as arg

def touch_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write('')

def test_attribute_getter():
    from collections import namedtuple
    gas = graph.get_attribute_stats

    Test = namedtuple('Test', 'x y z')

    x = 1
    y = 2
    z = 3

    test = Test(x, y, z)
    result = [test, test, test, test, test]
    results = [result, result, result, result, result]

    expected_mean = [1, 1, 1, 1, 1]
    expected_err = [0, 0, 0, 0, 0]

    # Test on all attributes in tuple

    mean, err = gas(results, 'x')
    assert mean == expected_mean
    assert err == expected_err

    expected_mean = [2, 2, 2, 2, 2]
    mean, err = gas(results, 'y')
    assert mean == expected_mean
    assert err == expected_err

    expected_mean = [3, 3, 3, 3, 3]
    mean, err = gas(results, 'z')
    assert mean == expected_mean
    assert err == expected_err

    # Trying with different attribute values

    x = [0, 1, 0, 1, 0, 1]
    y = [1, 2, 1, 2, 1, 2]
    z = [2, 3, 2, 3, 2, 3]

    test = Test(x, y, z)
    result = [test, test, test, test, test]
    results = [result, result, result, result, result]

    expected_mean = [.5, .5, .5, .5, .5]
    expected_err = [.5, .5, .5, .5, .5]
    mean, err = gas(results, 'x')

    assert mean == expected_mean
    assert err == expected_err

    expected_mean = [1.5, 1.5, 1.5, 1.5, 1.5]
    mean, err = gas(results, 'y')

    assert mean == expected_mean
    assert err == expected_err

    expected_mean = [2.5, 2.5, 2.5, 2.5, 2.5]
    mean, err = gas(results, 'z')

    assert mean == expected_mean
    assert err == expected_err

    # Test non-list values for results
    with pytest.raises(TypeError):
        gas(str(), 'x')

    with pytest.raises(TypeError):
        gas(set(), 'x')

    with pytest.raises(TypeError):
        gas(int(), 'x')

    # Test empty list for results
    with pytest.raises(AttributeError):
        gas([], 'x')

    # Test non-string values for attribute
    with pytest.raises(TypeError):
        gas(results, int())

    with pytest.raises(TypeError):
        gas(results, set())

    with pytest.raises(TypeError):
        gas(results, list())

    with pytest.raises(TypeError):
        gas(results, dict())

    # Test empty string for attribute
    with pytest.raises(AttributeError):
        gas(results, '')

def test_comparative_filename_retrieval():
    rmcf = graph.retrieve_method_comparative_filenames

    corpus = 'notacorpus'
    model = 'svm'
    size = 80
    methods = ['lc']

    results_dir = arg.results_dir(corpus)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Test base file retrieval
    base_file = os.path.join(results_dir, f'{corpus}_{model}_corpus{size}_results.pickle')
    touch_file(base_file)

    assert rmcf(corpus, [], model, size) == [base_file]

    # Test base and extra file
    lower = os.path.join(results_dir, f'{corpus}_lower_{model}_corpus{size}_results.pickle')
    touch_file(lower)

    assert rmcf(corpus, methods, model, size) == [base_file, lower]

    # Test double file retrieval with base
    methods = ['lc', 'np']

    nopunct = os.path.join(results_dir, f'{corpus}_nopunct_{model}_corpus{size}_results.pickle')
    touch_file(nopunct)

    assert rmcf(corpus, methods, model, size) == [base_file, lower, nopunct]

    # Test correct retrieval with irrelevant file
    irrelevant = os.path.join(results_dir, f'{corpus}_taters_{model}_corpus{size}_results.pickle')
    touch_file(irrelevant)

    assert rmcf(corpus, methods, model, size) == [base_file, lower, nopunct]

    # Test retrieving sample files
    sample_dir = os.path.join(os.getenv('HOME'), f'.preprocess/samples')
    corpus_sample_dir = os.path.join(sample_dir, f'{corpus}')

    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    if not os.path.isdir(corpus_sample_dir):
        os.mkdir(corpus_sample_dir)

    # We test less extensively because the code change is minimal between getting sample files
    # and results file
    base_sample = os.path.join(corpus_sample_dir, f'{corpus}_sample.pickle')
    lower_sample = os.path.join(corpus_sample_dir, f'{corpus}_lower_sample.pickle')
    nopunct_sample = os.path.join(corpus_sample_dir, f'{corpus}_nopunct_sample.pickle')
    irrelevant_sample = os.path.join(corpus_sample_dir, f'{corpus}_taters_sample.pickle')

    touch_file(base_sample)
    touch_file(lower_sample)
    touch_file(nopunct_sample)
    touch_file(irrelevant_sample)

    assert rmcf(corpus, methods, sample=True) == [base_sample, lower_sample, nopunct_sample]

    # Test non-strings for corpus
    with pytest.raises(TypeError):
        rmcf(int(), methods, model, size)

    with pytest.raises(TypeError):
        rmcf(set(), methods, model, size)

    with pytest.raises(TypeError):
        rmcf(dict(), methods, model, size)

    with pytest.raises(TypeError):
        rmcf(list(), methods, model, size)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        rmcf(str(), methods, model, size)

    # Test non-list types for methods
    with pytest.raises(TypeError):
        rmcf(corpus, str(), model, size)

    with pytest.raises(TypeError):
        rmcf(corpus, dict(), model, size)

    with pytest.raises(TypeError):
        rmcf(corpus, set(), model, size)

    with pytest.raises(TypeError):
        rmcf(corpus, int(), model, size)

    # Test non-strings for model
    with pytest.raises(TypeError):
        rmcf(corpus, methods, int(), size)

    with pytest.raises(TypeError):
        rmcf(corpus, methods, dict(), size)

    with pytest.raises(TypeError):
        rmcf(corpus, methods, set(), size)

    with pytest.raises(TypeError):
        rmcf(corpus, methods, list(), size)

    # Test empty string for model
    with pytest.raises(AttributeError):
        rmcf(corpus, methods, str(), size)

    # Test non-int types for size
    with pytest.raises(TypeError):
        rmcf(corpus, methods, model, str())

    with pytest.raises(TypeError):
        rmcf(corpus, methods, model, dict())

    with pytest.raises(TypeError):
        rmcf(corpus, methods, model, set())

    with pytest.raises(TypeError):
        rmcf(corpus, methods, model, list())

    # Test non-positive values for size
    with pytest.raises(AttributeError):
        rmcf(corpus, methods, model, 0)

    with pytest.raises(AttributeError):
        rmcf(corpus, methods, model, -1)

    with pytest.raises(AttributeError):
        rmcf(corpus, methods, model, -10000)

    os.remove(base_file)
    os.remove(lower)
    os.remove(nopunct)
    os.remove(irrelevant)
    os.remove(base_sample)
    os.remove(lower_sample)
    os.remove(nopunct_sample)
    os.remove(irrelevant_sample)
    os.rmdir(results_dir)
    os.rmdir(corpus_sample_dir)
    os.rmdir(sample_dir)

    assert not os.path.isfile(base_file)
    assert not os.path.isfile(lower)
    assert not os.path.isfile(nopunct)
    assert not os.path.isfile(irrelevant)
    assert not os.path.isfile(base_sample)
    assert not os.path.isfile(lower_sample)
    assert not os.path.isfile(nopunct_sample)
    assert not os.path.isfile(irrelevant_sample)
    assert not os.path.isdir(results_dir)
    assert not os.path.isdir(corpus_sample_dir)
    assert not os.path.isdir(sample_dir)

def test_progressive_filename_retrieval():
    rmpf = graph.retrieve_method_progressive_filenames

    # Test one filename retrieval
    corpus = 'notacorpus'
    model = 'svm'
    size = 80
    methods = ['lc']

    results_dir = arg.results_dir(corpus)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Test base file retrieval
    base_file = os.path.join(results_dir, f'{corpus}_{model}_corpus{size}_results.pickle')
    touch_file(base_file)

    assert rmpf(corpus, [], model, size) == [base_file]

    # Test retrieval with one method
    lower_file = os.path.join(results_dir, f'{corpus}_lower_{model}_corpus{size}_results.pickle')
    touch_file(lower_file)

    assert rmpf(corpus, methods, model, size) == [base_file, lower_file]

    # Test retrieval with multiple methods
    methods = ['lc', 'np']

    nopunct = os.path.join(results_dir, f'{corpus}_nopunct_{model}_corpus{size}_results.pickle')
    lower_nopunct = os.path.join(results_dir, f'{corpus}_lower_nopunct_{model}_corpus{size}_results.pickle')
    touch_file(nopunct)
    touch_file(lower_nopunct)

    # This also tests correct ordering
    # Also checks that the lower file is NOT grabbed
    assert rmpf(corpus, methods, model, size) == [base_file, nopunct, lower_nopunct]

    # Test correct retrieval with irrelevant file
    irrelevant_file = os.path.join(results_dir, f'{corpus}_taters_{model}_corpus{size}_results.pickle')
    touch_file(irrelevant_file)

    assert rmpf(corpus, methods, model, size) == [base_file, nopunct, lower_nopunct]

    # Test retrieving sample files
    sample_dir = os.path.join(os.getenv('HOME'), f'.preprocess/samples')
    corpus_sample_dir = os.path.join(sample_dir, f'{corpus}')

    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    if not os.path.isdir(corpus_sample_dir):
        os.mkdir(corpus_sample_dir)

    # We test less extensively because the code change is minimal between getting sample files
    # and results file
    base_sample = os.path.join(corpus_sample_dir, f'{corpus}_sample.pickle')
    nopunct_sample = os.path.join(corpus_sample_dir, f'{corpus}_nopunct_sample.pickle')
    nopunct_lower_sample = os.path.join(corpus_sample_dir, f'{corpus}_lower_nopunct_sample.pickle')
    irrelevant_sample = os.path.join(corpus_sample_dir, f'{corpus}_taters_sample.pickle')

    touch_file(base_sample)
    touch_file(nopunct_sample)
    touch_file(nopunct_lower_sample)
    touch_file(irrelevant_sample)

    assert rmpf(corpus, methods, sample=True) == [base_sample, nopunct_sample, nopunct_lower_sample]

    # Test non-strings for corpus
    with pytest.raises(TypeError):
        rmpf(int(), methods, model, size)

    with pytest.raises(TypeError):
        rmpf(set(), methods, model, size)

    with pytest.raises(TypeError):
        rmpf(dict(), methods, model, size)

    with pytest.raises(TypeError):
        rmpf(list(), methods, model, size)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        rmpf(str(), methods, model, size)

    # Test non-list types for methods
    with pytest.raises(TypeError):
        rmpf(corpus, str(), model, size)

    with pytest.raises(TypeError):
        rmpf(corpus, dict(), model, size)

    with pytest.raises(TypeError):
        rmpf(corpus, set(), model, size)

    with pytest.raises(TypeError):
        rmpf(corpus, int(), model, size)

    # Test non-strings for model
    with pytest.raises(TypeError):
        rmpf(corpus, methods, int(), size)

    with pytest.raises(TypeError):
        rmpf(corpus, methods, dict(), size)

    with pytest.raises(TypeError):
        rmpf(corpus, methods, set(), size)

    with pytest.raises(TypeError):
        rmpf(corpus, methods, list(), size)

    # Test emtpy string for model
    with pytest.raises(AttributeError):
        rmpf(corpus, methods, str(), size)

    # Test non-int types for size
    with pytest.raises(TypeError):
        rmpf(corpus, methods, model, str())

    with pytest.raises(TypeError):
        rmpf(corpus, methods, model, dict())

    with pytest.raises(TypeError):
        rmpf(corpus, methods, model, set())

    with pytest.raises(TypeError):
        rmpf(corpus, methods, model, list())

    # Test non-positive values for size
    with pytest.raises(AttributeError):
        rmpf(corpus, methods, model, 0)

    with pytest.raises(AttributeError):
        rmpf(corpus, methods, model, -1)

    with pytest.raises(AttributeError):
        rmpf(corpus, methods, model, -10000)

    os.remove(base_file)
    os.remove(lower_file)
    os.remove(nopunct)
    os.remove(lower_nopunct)
    os.remove(irrelevant_file)
    os.remove(base_sample)
    os.remove(nopunct_sample)
    os.remove(nopunct_lower_sample)
    os.remove(irrelevant_sample)
    os.rmdir(corpus_sample_dir)
    os.rmdir(sample_dir)
    os.rmdir(results_dir)

    assert not os.path.isfile(base_file)
    assert not os.path.isfile(lower_file)
    assert not os.path.isfile(nopunct)
    assert not os.path.isfile(lower_nopunct)
    assert not os.path.isfile(irrelevant_file)
    assert not os.path.isfile(base_sample)
    assert not os.path.isfile(nopunct_sample)
    assert not os.path.isfile(nopunct_lower_sample)
    assert not os.path.isfile(irrelevant_sample)
    assert not os.path.isdir(corpus_sample_dir)
    assert not os.path.isdir(sample_dir)
    assert not os.path.isdir(results_dir)

def test_filename_over_training_size_retrieval():
    rfots = graph.retrieve_filenames_over_training_size

    # Test one filename retrieval
    corpus = 'notacorpus'
    model = 'svm'
    results_dir = arg.results_dir(corpus)
    methods = ['lc']

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    expected_filename = os.path.join(results_dir, f'{corpus}_lower_{model}_corpus80_results.pickle')
    touch_file(expected_filename)

    assert rfots(corpus, methods, model) == [expected_filename]

    # Test one filename retrieval with irrelevant file thrown in
    irrelevant_file = os.path.join(results_dir, f'{corpus}_taters_{model}_corpus80_results.pickle')
    touch_file(irrelevant_file)

    assert rfots(corpus, methods, model) == [expected_filename]

    # Test multiple file retrieval with irrelevant file thrown in
    other_file = os.path.join(results_dir, f'{corpus}_lower_{model}_corpus800_results.pickle')
    touch_file(other_file)

    assert rfots(corpus, methods, model) == [expected_filename, other_file]

    # Test with badly formed corpus tag
    bad_file = os.path.join(results_dir, f'{corpus}_lower_{model}_corpuscats_results.pickle')
    touch_file(bad_file)

    assert rfots(corpus, methods, model) == [expected_filename, other_file]

    # Test 3 files with irrelevant file thrown in
    other_other_file = os.path.join(results_dir, f'{corpus}_lower_{model}_corpus774_results.pickle')
    touch_file(other_other_file)

    assert rfots(corpus, methods, model) == [expected_filename, other_file, other_other_file]

    # Test non-strings for corpus
    with pytest.raises(TypeError):
        rfots(int(), methods, model)

    with pytest.raises(TypeError):
        rfots(set(), methods, model)

    with pytest.raises(TypeError):
        rfots(dict(), methods, model)

    with pytest.raises(TypeError):
        rfots(list(), methods, model)

    # Test empty string for corpus
    with pytest.raises(AttributeError):
        rfots(str(), methods, model)

    # Test non-list types for methods
    with pytest.raises(TypeError):
        rfots(corpus, str(), model)

    with pytest.raises(TypeError):
        rfots(corpus, dict(), model)

    with pytest.raises(TypeError):
        rfots(corpus, set(), model)

    with pytest.raises(TypeError):
        rfots(corpus, int(), model)

    # Test non-strings for model
    with pytest.raises(TypeError):
        rfots(corpus, methods, int())

    with pytest.raises(TypeError):
        rfots(corpus, methods, dict())

    with pytest.raises(TypeError):
        rfots(corpus, methods, set())

    with pytest.raises(TypeError):
        rfots(corpus, methods, list())

    # Test emtpy string for model
    with pytest.raises(AttributeError):
        rfots(corpus, methods, str())

    # Cleanup
    os.remove(expected_filename)
    os.remove(irrelevant_file)
    os.remove(other_file)
    os.remove(bad_file)
    os.remove(other_other_file)
    os.rmdir(results_dir)

    assert not os.path.isfile(expected_filename)
    assert not os.path.isfile(irrelevant_file)
    assert not os.path.isfile(other_file)
    assert not os.path.isfile(bad_file)
    assert not os.path.isfile(other_other_file)
    assert not os.path.isdir(results_dir)
