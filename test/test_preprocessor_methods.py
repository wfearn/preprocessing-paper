from preprocess import argmanager
import itertools
import pytest

def test_import_corpus():
    from preprocess import preprocessor
    import os
    import gzip
    import difflib

    base_name = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon.json.gz'
    f = preprocessor.preprocess_corpus('testamazon', [])
    assert f.name == base_name
    assert type(f) is gzip.GzipFile
    f.close()

    lower_name = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_lower.json.gz'
    with gzip.open(lower_name, 'wb') as f:
        f.write('stuff'.encode('utf-8'))
    assert os.path.isfile(lower_name)

    f = preprocessor.preprocess_corpus('testamazon', ['lc'])
    assert f.name == lower_name
    assert type(f) is gzip.GzipFile
    f.close()
    os.remove(lower_name)

    assert not os.path.isfile(lower_name)

    with pytest.raises(AttributeError):
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'])
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'])
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'])
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'], offset=0)
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'], offset=0, run_number=6)
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'], run_number=6)
        preprocessor.preprocess_corpus('notacorpus', ['lc', 'sc'], offset=6)

        preprocessor.preprocess_corpus('testamazon', ['lc', 'sc'])
        preprocessor.preprocess_corpus('testamazon', ['lc'])
        preprocessor.preprocess_corpus('testamazon', ['sc'])

        preprocessor.preprocess_corpus('testamazon', [], offset=0)
        preprocessor.preprocess_corpus('testamazon', [], offset=-1)
        preprocessor.preprocess_corpus('testamazon', [], offset=-2)

        preprocessor.preprocess_corpus('testamazon', [], run_number=-1)
        preprocessor.preprocess_corpus('testamazon', [], run_number=-2)

    run_number = 10000
    offset = 6
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_test_10000.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    with gzip.open(testfile, 'rb') as f:
        for i, line in enumerate(f):
            # there should only be one line in the file
            assert not i

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    run_number = 0
    offset = 1
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_test_0.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = '{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}\n'

    notexpected = 'anything else'

    with gzip.open(testfile, 'rb') as f:
        for i, line in enumerate(f):
            # there should only be one line in the file
            assert not i
            l = line.decode('utf-8')
            assert l != notexpected
            assert l == expected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    # Test to make sure stopword removal is put in file name but does nothing

    run_number = 0
    offset = 1
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_stop_0.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['sp'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = '{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}\n'

    notexpected = 'anything else'

    with gzip.open(testfile, 'rb') as f:
        for i, line in enumerate(f):
            # there should only be one line in the file
            assert not i
            l = line.decode('utf-8')
            assert l != notexpected
            assert l == expected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_lower_test_0.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['lc', 'tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = '{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"conversations with god book 1 is the single most extraordinary book i have ever read!!!it totally changed my life. i would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. this book did wonders for my relationship with god, myself and everyone around me. i approach living differently, i enjoy life more. i have had a copy of this book since it was first published (1997)? and i still turn to it again and again for spiritual enlightenment, upliftment and remembering.i love this book and i love neale walsch for his courage in writing it. unbelievable! a must read!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}\n'

    notexpected = 'anything else'

    with gzip.open(testfile, 'rb') as f:
        for i, line in enumerate(f):
            # there should only be one line in the file
            assert not i
            l = line.decode('utf-8')
            assert l != notexpected
            assert l == expected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    run_number = 5
    offset = 1
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_test_5.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = '{\"reviewerID\": \"A3UTQPQPM4TQO0\", \"asin\": \"0000013714\", \"reviewerName\": \"betty burnett\", \"helpful\": [0, 0], \"reviewText\": \"We have many of the old, old issue. But the number had depleted. There were not enough books to allow us to use them regularly. With the additional supply the books will be used more often. They arre a good old standby for gospel singing.\", \"overall\": 5.0, \"summary\": \"I was disappointed that you would only allow me to purchase 4 when your inventory showed that you had 14 available.\", \"unixReviewTime\": 1374883200, \"reviewTime\": \"07 27, 2013\"}\n'

    notexpected = 'anything else'

    with gzip.open(testfile, 'rb') as f:
        for i, line in enumerate(f):
            # there should only be one line in the file
            assert not i
            l = line.decode('utf-8')
            assert l != notexpected
            assert l == expected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    run_number = 0
    offset = 2
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_lower_test_0.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['lc', 'tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = """{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"conversations with god book 1 is the single most extraordinary book i have ever read!!!it totally changed my life. i would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. this book did wonders for my relationship with god, myself and everyone around me. i approach living differently, i enjoy life more. i have had a copy of this book since it was first published (1997)? and i still turn to it again and again for spiritual enlightenment, upliftment and remembering.i love this book and i love neale walsch for his courage in writing it. unbelievable! a must read!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}
{\"reviewerID\": \"AH2L9G3DQHHAJ\", \"asin\": \"0000000116\", \"reviewerName\": \"chris\", \"helpful\": [5, 5], \"reviewText\": \"interesting grisham tale of a lawyer that takes millions of dollars from his firm after faking his own death. grisham usually is able to hook his readers early and ,in this case, doesn't play his hand to soon. the usually reliable frank mueller makes this story even an even better bet on audiobook.\", \"overall\": 4.0, \"summary\": \"Show me the money!\", \"unixReviewTime\": 1019865600, \"reviewTime\": \"04 27, 2002\"}\n"""

    notexpected = 'anything_else'

    with gzip.open(testfile, 'rb') as f:
        text = ''.join([l.decode('utf-8') for l in f.readlines()])

    assert text == expected
    assert text != notexpected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    run_number = 1
    offset = 2
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_lower_test_1.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['lc', 'tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)


    expected = """{\"reviewerID\": \"A2IIIDRK3PRRZY\", \"asin\": \"0000000116\", \"reviewerName\": \"Helene\", \"helpful\": [0, 0], \"reviewText\": \"the thumbnail is a shirt. the product shown is a shoe. the description is a book.this reviewer is confused.\", \"overall\": 1.0, \"summary\": \"Listing is all screwed up\", \"unixReviewTime\": 1395619200, \"reviewTime\": \"03 24, 2014\"}
{\"reviewerID\": \"A1TADCM7YWPQ8M\", \"asin\": \"0000000868\", \"reviewerName\": \"Joel@AWS\", \"helpful\": [10, 10], \"reviewText\": \"i'll be honest. i work for a large online retailer named after a south american river, and ordered this because the asin (0000000868) is so ... distinctive ... and, frankly, i wondered what the hell a &quot;reader turntable&quot; was, particularly one with both an editor and an illustrator.a friend suggested that the title was really the result of a very poor translation; that &quot;reader&quot; was supposed to be &quot;tales&quot;, &quot;turntable&quot; was supposed to be &quot;lyrics&quot; and this was really a book entitled &quot;korea's favorite tales and lyrics&quot;. oddly enough, there is such a book, apparently edited by mr. hyun and illustrated by mr. park.well, as it turns out, &quot;reader turntable se-14&quot; is not a book. it is, well, a reader turntable: a black rectangular piece of 1/2&quot; particleboard, about 13x16 inches, mounted on a bearing so it can rotate easily. it's a lot like a &quot;lazy susan&quot;.it's useful for reading rooms, libraries, etc. place it on table, put a dictionary or other big heavy reference book on it, and now folks sitting on either side of the table can share the reference, simply turning it towards them when they need to look something up.construction is sturdy. i (175 lbs) stood on it and it rotated: a bit roughly but it didn't collapse. bearing quality is modest but serviceable. black matte finish is tasteful.\", \"overall\": 4.0, \"summary\": \"Not a Bad Translation\", \"unixReviewTime\": 1031702400, \"reviewTime\": \"09 11, 2002\"}\n"""

    notexpected = 'anything_else'

    with gzip.open(testfile, 'rb') as f:
        text = ''.join([l.decode('utf-8') for l in f.readlines()])

    assert text == expected
    assert text != notexpected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    # Test on reddit data
    run_number = 0
    offset = 1
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testreddit/testreddit_test_0.json.gz'
    preprocessor.preprocess_corpus('testreddit', ['tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)

    expected = """{"author": "nanakishi", "author_flair_css_class": "Owner", "author_flair_text": "C4 Head", "body": "/u/azeleon", "can_gild": true, "controversiality": 0, "created_utc": 1496275200, "distinguished": null, "edited": false, "gilded": 0, "id": "diandlp", "link_id": "t3_6ej12i", "parent_id": "t3_6ej12i", "retrieved_on": 1498961706, "score": 2, "stickied": false, "subreddit": "RPGStuck_C4", "subreddit_id": "t5_3e5e1"}\n"""

    notexpected = 'anything_else'

    with gzip.open(testfile, 'rb') as f:
        text = ''.join([l.decode('utf-8') for l in f.readlines()])

    assert text == expected
    assert text != notexpected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    run_number = 1
    offset = 10000
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testamazon/testamazon_lower_test_1.json.gz'
    preprocessor.preprocess_corpus('testamazon', ['lc', 'tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)


    expected = ''
    notexpected = 'anything_else'

    with gzip.open(testfile, 'rb') as f:
        text = ''.join([l.decode('utf-8') for l in f.readlines()])

    assert text == expected
    assert text != notexpected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

    base_name = os.getenv('HOME') + '/.preprocess/corpora/testapnews/testapnews.txt.gz'
    f = preprocessor.preprocess_corpus('testapnews', [])
    assert f.name == base_name
    assert type(f) is gzip.GzipFile
    f.close()

    lower_name = os.getenv('HOME') + '/.preprocess/corpora/testapnews/testapnews_lower.txt.gz'
    with gzip.open(lower_name, 'wb') as f:
        f.write('stuff'.encode('utf-8'))
    assert os.path.isfile(lower_name)

    f = preprocessor.preprocess_corpus('testapnews', ['lc'])
    assert f.name == lower_name
    assert type(f) is gzip.GzipFile
    f.close()
    os.remove(lower_name)

    assert not os.path.isfile(lower_name)

    run_number = 0
    offset = 2
    testfile = os.getenv('HOME') + '/.preprocess/corpora/testapnews/testapnews_lower_test_0.txt.gz'
    preprocessor.preprocess_corpus('testapnews', ['lc', 'tt'], run_number=run_number, offset=offset)
    assert os.path.isfile(testfile)


    expected = """boston (ap) &md; an endowed chair has been named at a northern ireland university program for thomas p. ``tip'' o'neill, the late speaker of the u.s. house of representatives.
``the american ireland fund is proud to establish this honorable and prestigious post in memory of tip o'neill for his spirited dedication to resolving the differences in northern ireland without violence,'' kingsley aikins, executive director of the american ireland fund, said thursday.\n"""
    notexpected = 'anything_else'

    with gzip.open(testfile, 'rb') as f:
        text = ''.join([l.decode('utf-8') for l in f.readlines()])

    assert text == expected
    assert text != notexpected

    os.remove(testfile)
    assert not os.path.isfile(testfile)

def test_retrieve_corpus_file():
    import os

    rcf = argmanager.retrieve_corpus_file

    methods = ['sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_spell.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['ws', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_spell_seg.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['st', 'sp', 'nr', 'ud', 'np', 'lc', 'ws', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_spell_seg_udrep_nrem_stop_stem.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['lc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_lower.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['lc', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_spell.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['lc', 'ws']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_seg.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['lc', 'ws', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/reddit/reddit_spell_seg.json.gz')
    assert rcf('reddit', methods) == expected

    methods = ['lc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_lower.txt.gz')
    assert rcf('apnews', methods) == expected

    methods = ['lc', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_spell.txt.gz')
    assert rcf('apnews', methods) == expected

    methods = ['lc', 'ws']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_seg.txt.gz')
    assert rcf('apnews', methods) == expected

    methods = ['lc', 'ws', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_spell_seg.txt.gz')
    assert rcf('apnews', methods) == expected

    methods = ['lc', 'ws']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_seg_0.txt.gz')
    assert rcf('apnews', methods, run_number=0) == expected

    methods = ['lc', 'ws', 'sc']
    expected = os.path.join(os.path.join(os.getenv('HOME'), '.preprocess'), 'corpora/apnews/apnews_spell_seg_66.txt.gz')
    assert rcf('apnews', methods, run_number=66) == expected

def test_processed_corpus_name():
    cn = argmanager.processed_corpus_name

    methods = ['sc']
    expected = 'reddit_spell'
    assert cn('reddit', methods) == expected

    methods = ['ws', 'sc']
    expected = 'reddit_spell_seg'
    assert cn('reddit', methods) == expected

    methods = ['st', 'sp', 'nr', 'ud', 'np', 'lc', 'ws', 'sc']
    expected = 'reddit_spell_seg_udrep_nrem_stop_stem'
    assert cn('reddit', methods) == expected

    methods = ['lc']
    expected = 'reddit_lower'
    assert cn('reddit', methods) == expected

    methods = ['lc', 'sc']
    expected = 'reddit_spell'
    assert cn('reddit', methods) == expected

    methods = ['lc', 'ws']
    expected = 'reddit_seg'
    assert cn('reddit', methods) == expected

    methods = ['lc', 'ws', 'sc']
    expected = 'reddit_spell_seg'
    assert cn('reddit', methods) == expected

    methods = ['np', 'ws', 'sc']
    expected = 'reddit_spell_seg'
    assert cn('reddit', methods) == expected

    methods = ['lc']
    expected = 'apnews_lower'
    assert cn('apnews', methods) == expected

    methods = ['lc', 'sc']
    expected = 'apnews_spell'
    assert cn('apnews', methods) == expected

    methods = ['lc', 'ws']
    expected = 'apnews_seg'
    assert cn('apnews', methods) == expected

    methods = ['np', 'ws']
    expected = 'apnews_seg'
    assert cn('apnews', methods) == expected

    methods = ['lc', 'ws', 'sc']
    expected = 'apnews_spell_seg'
    assert cn('apnews', methods) == expected

def test_sort_methods():
    from preprocess import methods

    sm = methods.sort_methods

    # Test empty list
    methods = []
    assert sm(methods) == methods

    # Test one element list
    methods = ['np']
    assert sm(methods) == methods

    # Test two element list with no priority
    methods = ['nr', 'np']
    assert sm(methods) == methods

    # Test keeping order for two elements with priority
    methods = ['np', 'ws']
    assert sm(methods) == methods

    methods = ['np', 'sc']
    assert sm(methods) == methods

    methods = ['np', 'st']
    assert sm(methods) == methods

    methods = ['np', 'lc']
    assert sm(methods) == methods

    # Test imposing correct order for two elements with priority
    methods = ['ws', 'np']
    expected = ['np', 'ws']
    assert sm(methods) == expected

    methods = ['sc', 'np']
    expected = ['np', 'sc']
    assert sm(methods) == expected

    methods = ['st', 'np']
    expected = ['np', 'st']
    assert sm(methods) == expected

    methods = ['lc', 'np']
    expected = ['np', 'lc']
    assert sm(methods) == expected

    # Test correct order for many elements out of order
    methods = ['lc', 'st', 'sc', 'ws', 'ud']
    expected = ['ud', 'ws', 'sc', 'st', 'lc']
    assert sm(methods) == expected

    # Test non-list types
    with pytest.raises(TypeError):
        sm(int())

    with pytest.raises(TypeError):
        sm(dict())

    with pytest.raises(TypeError):
        sm(set())

    with pytest.raises(TypeError):
        sm(tuple((0, 0)))
