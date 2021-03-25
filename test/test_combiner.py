from preprocess import combiner
import gzip
import pytest
import pickle
from pathlib import Path
import os


def test_combine_pickles():

    cp = combiner.combine_pickles

    files = ['testpickle_1.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']
    for pick in files:
        with open(pick, 'wb') as f:
            pickle.dump(tuple((0, 0)), f)

    assert not os.path.isfile('testpickle.pickle')
    cp('testpickle', files)

    with open('testpickle.pickle', 'rb') as f:
        results = pickle.load(f)

    os.remove('testpickle.pickle')

    assert results == [(0, 0), (0, 0), (0, 0)]
    for f in files:
        assert not os.path.isfile(f)

    testdata = [30, 'string', tuple((0, 0)), { 'Key', 'Value' }]
    for pick in files:
        with open(pick, 'wb') as f:
            pickle.dump(testdata, f)

    assert not os.path.isfile('testpickle.pickle')
    cp('testpickle', files)

    with open('testpickle.pickle', 'rb') as f:
        results = pickle.load(f)

    os.remove('testpickle.pickle')

    from copy import deepcopy
    for result in results:
        for i, data in enumerate(result):
            assert data == result[i]

    with pytest.raises(ValueError):
        cp('francis', files)

        files = ['michael.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']
        cp('testpickle', files)

        files = ['testpickle_1.pickle', 'finkle_2.pickle', 'testpickle_3.pickle']
        cp('testpickle', files)


def test_combine_corpora():

    cc = combiner.combine_corpora

    # Test json extension

    files = ['testfile_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']
    for f in files:
        Path(f).touch()

    cc('testfile', files)

    expected_file = 'testfile.json.gz'
    assert os.path.isfile(expected_file)
    for f in files:
        assert not os.path.isfile(f)

    with gzip.open(expected_file, 'rb') as f:
        text = f.read().decode('utf-8')

    assert text == ''

    os.remove(expected_file)
    assert not os.path.isfile(expected_file)

    # Test txt extension

    files = ['testfile_1.txt.gz', 'testfile_2.txt.gz', 'testfile_3.txt.gz']
    for f in files:
        Path(f).touch()

    cc('testfile', files)

    expected_file = 'testfile.txt.gz'
    assert os.path.isfile(expected_file)
    for f in files:
        assert not os.path.isfile(f)

    with gzip.open(expected_file) as f:
        text = f.read().decode('utf-8')

    assert text == ''

    os.remove(expected_file)
    assert not os.path.isfile(expected_file)

    # Test information integrity for text files

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write('test\n'.encode('utf-8'))

    expected = 'test\ntest\ntest\n'

    cc('testfile', files)

    with gzip.open('testfile.txt.gz', 'rb') as f:
        text = f.read().decode('utf-8')

    assert text == expected
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')

    # Test real information integrity for text files

    teststring = """boston (ap) &md; an endowed chair has been named at a northern ireland university program for thomas p. ``tip'' o'neill, the late speaker of the u.s. house of representatives."""

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}\n'.encode('utf-8'))

    expected = f'{teststring}\n{teststring}\n{teststring}\n'

    cc('testfile', files)

    with gzip.open('testfile.txt.gz', 'rb') as f:
        text = f.read().decode('utf-8')

    assert text == expected
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')


    # Test information integrity for json files
    files = ['testfile_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']

    teststring = """{\"reviewerID\": \"A2IIIDRK3PRRZY\", \"asin\": \"0000000116\", \"reviewerName\": \"Helene\", \"helpful\": [0, 0], \"reviewText\": \"the thumbnail is a shirt. the product shown is a shoe. the description is a book.this reviewer is confused.\", \"overall\": 1.0, \"summary\": \"Listing is all screwed up\", \"unixReviewTime\": 1395619200, \"reviewTime\": \"03 24, 2014\"}"""

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}\n'.encode('utf-8'))

    cc('testfile', files)

    with gzip.open('testfile.json.gz', 'rb') as f:
        text = f.read().decode('utf-8')

    assert text == f'{teststring}\n{teststring}\n{teststring}\n'

    os.remove('testfile.json.gz')
    assert not os.path.isfile('testfile.json.gz')

    with pytest.raises(ValueError):
        files = ['crombopulous_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']
        cc('testfile', files)

        files = ['testfile_1.json.gz', 'testfile_2.json.gz', 'michael_3.json.gz']
        cc('testfile', files)

        files = ['testfile_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']
        cc('regolby', files)

def test_json_combiner():
    c = combiner.combine

    # Test for basic json.gz
    files = ['testfile_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']

    teststring = 'stuff'

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    for f in files:
        assert not os.path.isfile(f)

    assert os.path.isfile('testfile.json.gz')
    os.remove('testfile.json.gz')
    assert not os.path.isfile('testfile.json.gz')

    # Test different kinds of json.gz
    files = ['crombopulous_2.json.gz', 'crombopulous_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.json.gz')
    assert os.path.isfile('testfile.json.gz')
    os.remove('crombopulous.json.gz')
    os.remove('testfile.json.gz')
    assert not os.path.isfile('crombopulous.json.gz')
    assert not os.path.isfile('testfile.json.gz')

    # Test for only one json.gz of a particular category
    files = ['crombopulous_1.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.json.gz')
    assert os.path.isfile('testfile.json.gz')
    os.remove('crombopulous.json.gz')
    os.remove('testfile.json.gz')
    assert not os.path.isfile('crombopulous.json.gz')
    assert not os.path.isfile('testfile.json.gz')

    # Test to make sure non-number files are untouched
    files = ['crombopulous.json.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    first_mod_time = os.path.getmtime('crombopulous.json.gz')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.json.gz')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.json.gz')
    os.remove('crombopulous.json.gz')
    assert not os.path.isfile('crombopulous.json.gz')

    # Test to make sure non-number files are untouched while numbered files are
    files = ['crombopulous.json.gz', 'testfile_2.json.gz', 'testfile_3.json.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    first_mod_time = os.path.getmtime('crombopulous.json.gz')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.json.gz')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.json.gz')
    assert os.path.isfile('testfile.json.gz')
    os.remove('crombopulous.json.gz')
    os.remove('testfile.json.gz')
    assert not os.path.isfile('crombopulous.json.gz')
    assert not os.path.isfile('testfile.json.gz')

    # Test for functionality in a directory that isn't cwd
    os.mkdir(os.path.join(os.getcwd(), 'test'))
    files = ['test/testfile_1.json.gz', 'test/testfile_2.json.gz', 'test/testfile_3.json.gz']

    teststring = 'stuff'

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.path.join(os.getcwd(), 'test'))

    for f in files:
        assert not os.path.isfile(f)

    assert os.path.isfile('test/testfile.json.gz')
    os.remove('test/testfile.json.gz')
    assert not os.path.isfile('test/testfile.json.gz')

    os.rmdir(os.path.join(os.getcwd(), 'test'))

def test_text_combiner():
    c = combiner.combine

    # Test for basic txt.gz
    files = ['testfile_1.txt.gz', 'testfile_2.txt.gz', 'testfile_3.txt.gz']

    teststring = 'stuff'

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    for f in files:
        assert not os.path.isfile(f)

    assert os.path.isfile('testfile.txt.gz')
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')

    # Test different kinds of txt.gz
    files = ['crombopulous_2.txt.gz', 'crombopulous_1.txt.gz', 'testfile_2.txt.gz', 'testfile_3.txt.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.txt.gz')
    assert os.path.isfile('testfile.txt.gz')
    os.remove('crombopulous.txt.gz')
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('crombopulous.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')

    # Test for only one txt.gz of a particular category
    files = ['crombopulous_1.txt.gz', 'testfile_2.txt.gz', 'testfile_3.txt.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.txt.gz')
    assert os.path.isfile('testfile.txt.gz')
    os.remove('crombopulous.txt.gz')
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('crombopulous.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')

    # Test to make sure non-number files are untouched
    files = ['crombopulous.txt.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    first_mod_time = os.path.getmtime('crombopulous.txt.gz')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.txt.gz')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.txt.gz')
    os.remove('crombopulous.txt.gz')
    assert not os.path.isfile('crombopulous.txt.gz')

    # Test to make sure non-number files are untouched while numbered files are
    files = ['crombopulous.txt.gz', 'testfile_2.txt.gz', 'testfile_3.txt.gz']


    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    first_mod_time = os.path.getmtime('crombopulous.txt.gz')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.txt.gz')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.txt.gz')
    assert os.path.isfile('testfile.txt.gz')
    os.remove('crombopulous.txt.gz')
    os.remove('testfile.txt.gz')
    assert not os.path.isfile('crombopulous.txt.gz')
    assert not os.path.isfile('testfile.txt.gz')

    # Test for functionality in a directory that isn't cwd
    os.mkdir(os.path.join(os.getcwd(), 'test'))
    files = ['test/testfile_1.txt.gz', 'test/testfile_2.txt.gz', 'test/testfile_3.txt.gz']

    teststring = 'stuff'

    for f in files:
        with gzip.open(f, 'wb') as tf:
            tf.write(f'{teststring}'.encode('utf-8'))
        assert os.path.isfile(f)

    c(os.path.join(os.getcwd(), 'test'))

    for f in files:
        assert not os.path.isfile(f)

    assert os.path.isfile('test/testfile.txt.gz')
    os.remove('test/testfile.txt.gz')
    assert not os.path.isfile('test/testfile.txt.gz')

    os.rmdir(os.path.join(os.getcwd(), 'test'))

def test_pickle_combiner():
    c = combiner.combine

    # Test for basic pickle
    files = ['testpickle_1.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']

    teststring = 'stuff'

    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    c(os.getcwd())

    for pick in files:
        assert not os.path.isfile(pick)

    assert os.path.isfile('testpickle.pickle')
    os.remove('testpickle.pickle')
    assert not os.path.isfile('testpickle.pickle')

    # Test different kinds of pickle
    files = ['crombopulous_2.pickle', 'crombopulous_1.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']


    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.pickle')
    assert os.path.isfile('testpickle.pickle')
    os.remove('crombopulous.pickle')
    os.remove('testpickle.pickle')
    assert not os.path.isfile('crombopulous.pickle')
    assert not os.path.isfile('testpickle.pickle')

    files = ['testpickle_1.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']
    for pick in files:
        with open(pick, 'wb') as f:
            pickle.dump(tuple((0, 0)), f)

    c(os.getcwd())

    assert os.path.isfile('testpickle.pickle')
    os.remove('testpickle.pickle')
    assert not os.path.isfile('testpickle.pickle')

    # Test for only one pickle of a particular category
    files = ['crombopulous_1.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']


    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    c(os.getcwd())

    assert os.path.isfile('crombopulous.pickle')
    assert os.path.isfile('testpickle.pickle')
    os.remove('crombopulous.pickle')
    os.remove('testpickle.pickle')
    assert not os.path.isfile('crombopulous.pickle')
    assert not os.path.isfile('testpickle.pickle')

    # Test to make sure non-number files are untouched
    files = ['crombopulous.pickle']


    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    first_mod_time = os.path.getmtime('crombopulous.pickle')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.pickle')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.pickle')
    os.remove('crombopulous.pickle')
    assert not os.path.isfile('crombopulous.pickle')

    # Test to make sure non-number files are untouched while numbered files are
    files = ['crombopulous.pickle', 'testpickle_2.pickle', 'testpickle_3.pickle']


    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    first_mod_time = os.path.getmtime('crombopulous.pickle')

    c(os.getcwd())

    second_mod_time = os.path.getmtime('crombopulous.pickle')
    assert first_mod_time == second_mod_time

    assert os.path.isfile('crombopulous.pickle')
    assert os.path.isfile('testpickle.pickle')
    os.remove('crombopulous.pickle')
    os.remove('testpickle.pickle')
    assert not os.path.isfile('crombopulous.pickle')
    assert not os.path.isfile('testpickle.pickle')

    # Test for functionality in a directory that isn't cwd
    os.mkdir(os.path.join(os.getcwd(), 'test'))
    files = ['test/testpickle_1.pickle', 'test/testpickle_2.pickle', 'test/testpickle_3.pickle']

    teststring = 'stuff'

    for pick in files:
        with open(pick, 'wb') as tf:
            pickle.dump(tuple((0, 0)), tf)
        assert os.path.isfile(pick)

    c(os.path.join(os.getcwd(), 'test'))

    for pick in files:
        assert not os.path.isfile(pick)

    assert os.path.isfile('test/testpickle.pickle')
    os.remove('test/testpickle.pickle')
    assert not os.path.isfile('test/testpickle.pickle')

    os.rmdir(os.path.join(os.getcwd(), 'test'))
