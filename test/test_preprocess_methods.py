from preprocess import methods
import itertools
import pytest

def test_json_processor():
    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    expected = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    je = methods.json_processor(methods.base_preprocessor(), tag='reviewText')
    assert je(s) == expected

    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    expected = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"conversations with god book 1 is the single most extraordinary book i have ever read!!!it totally changed my life. i would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. this book did wonders for my relationship with god, myself and everyone around me. i approach living differently, i enjoy life more. i have had a copy of this book since it was first published (1997)? and i still turn to it again and again for spiritual enlightenment, upliftment and remembering.i love this book and i love neale walsch for his courage in writing it. unbelievable! a must read!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    je = methods.json_processor(methods.lowercase_preprocessor(), tag='reviewText')
    assert je(s) == expected


def test_plaintext_processor():
    s = "{\"reviewerID\": \"A3AF8FFZAZYNE5\", \"asin\": \"0000000078\", \"helpful\": [1, 1], \"reviewText\": \"Conversations with God Book 1 is the single most extraordinary book I have ever read!!!It totally changed my life. I would recommend it to anyone who is seeking emotional and spiritual growth, freedom and empowerment. This book did wonders for my relationship with God, myself and everyone around me. I approach living differently, I enjoy life more. I have had a copy of this book since it was first published (1997)? and I still turn to it again and again for spiritual enlightenment, upliftment and remembering.I love this book and I love Neale Walsch for his courage in writing it. Unbelievable! A MUST READ!!!\", \"overall\": 5.0, \"summary\": \"Impactful!\", \"unixReviewTime\": 1092182400, \"reviewTime\": \"08 11, 2004\"}"

    pe = methods.plaintext_processor(methods.base_preprocessor())
    assert pe(s) == s

    pe = methods.plaintext_processor(methods.punctuation_preprocessor())
    result = "reviewerID A3AF8FFZAZYNE5 asin 0000000078 helpful 1 1 reviewText Conversations with God Book 1 is the single most extraordinary book I have ever readIt totally changed my life I would recommend it to anyone who is seeking emotional and spiritual growth freedom and empowerment This book did wonders for my relationship with God myself and everyone around me I approach living differently I enjoy life more I have had a copy of this book since it was first published 1997 and I still turn to it again and again for spiritual enlightenment upliftment and rememberingI love this book and I love Neale Walsch for his courage in writing it Unbelievable A MUST READ overall 50 summary Impactful unixReviewTime 1092182400 reviewTime 08 11 2004"
    assert pe(s) == result


def test_number_preprocessor():
    np = methods.number_preprocessor()

    s_pre = 'habbiddy jabbidjs akdjflska djfks a   a sjd a sjd      sjd jskj    jskd'
    assert np(s_pre) == s_pre

    s_pre = '6'
    assert np(s_pre) == ''

    s_pre = ''
    assert np(s_pre) == s_pre

    s_pre = 'w0rd   and w2134566rd and    %%%%%%##@@@@ 123556'
    s_post = 'wrd   and wrd and    %%%%%%##@@@@ '
    assert np(s_pre) == s_post

    s_pre = 'the mAn of username tyth3b0n3s upEnded the status of the game Tuesday'
    s_post = 'the mAn of username tythbns upEnded the status of the game Tuesday'
    assert np(s_pre) == s_post


def test_base_preprocessor():
    bp = methods.base_preprocessor()

    s_pre = 'word  word word word     word word word '
    s_post = 'word word word word word word word'

    assert bp(s_pre) == s_post
    assert bp(s_post) == s_post

    empty = ''
    assert bp(empty) == empty

    s_pre = ' word '
    s_post = 'word'
    assert bp(s_pre) == s_post
    assert bp(s_post) == s_post

    s_pre = 'Word  wOrd worD $5%sJJd '
    s_post = 'Word wOrd worD $5%sJJd'
    assert bp(s_pre) == s_post
    assert bp(s_post) == s_post

    s_pre = '   wor d  '
    s_post = 'wor d'
    assert bp(s_pre) == s_post
    assert bp(s_post) == s_post

def test_lemmatizing_preprocessor():
    lp = methods.lemmatizing_preprocessor()

    # Test base functionality
    pre = 'rocks'
    post = 'rock'

    assert lp(pre) == post

    pre = 'corpora'
    post = 'corpus'

    assert lp(pre) == post

    # Test more complicated cases
    pre = 'better'

    assert lp(pre) == pre

    pre = 'pontification'

    assert lp(pre) == pre

    # Test two words in a row
    pre = 'rocks corpora'
    post = 'rock corpus'

    assert lp(pre) == post

    # Test multiple words in a row
    pre = 'rocks corpora better gooder camels pontification'
    post = 'rock corpus better gooder camel pontification'

    assert lp(pre) == post

    # Test that lemmatized words don't get changed
    assert lp(post) == post

    # Test that numbers and punctuation aren't affected
    pre = '6373 1293948 193843 1092992 38348 ##$$##$%%##%#$$'
    assert lp(pre) == pre

def test_stemming_preprocessor():
    sp = methods.stemming_preprocessor()

    pre = 'wording caresses flies dies mules denied died agreed owned humbled sized'
    post = 'word caress fli die mule deni die agre own humbl size'
    postpost = 'word caress fli die mule deni die agr own humbl size'
    assert sp(pre) == post
    assert sp(post) == postpost

    pre = ''
    assert sp(pre) == pre

    pre = '6373 1293948 193843 1092992 38348 ##$$##$%%##%#$$'
    assert sp(pre) == pre


def test_punctuation_preprocessor():
    pp = methods.punctuation_preprocessor()

    pre = 'word word word word word word word'
    assert pp(pre) == pre

    pre = '!@#$%^&*()_+}{|":?><~'
    assert pp(pre) == ''
    assert pp('') == ''

    pre = 'word w0rD    37#$umbol   {}grand  ju":niper'
    post = 'word w0rD    37umbol   grand  juniper'
    assert pp(pre) == post
    assert pp(post) == post

def test_dash_underscore_preprocessor():
    du = methods.dash_underscore_preprocessor()

    pre = 'word_word_word_word-word-word-word'
    post = 'word word word word word word word'
    assert du(pre) == post

    pre = 'word_______word'
    post = 'word word'
    assert du(pre) == post

    pre = 'word-------word'
    post = 'word word'
    assert du(pre) == post

    pre = 'word_-_-_-_-_-word'
    post = 'word word'
    assert du(pre) == post

    pre = 'word_word-word__word--word___word---word'
    post = 'word word word word word word word'
    assert du(pre) == post

    pre = '1234word1234'
    assert du(pre) == pre

    pre = 'word111word'
    assert du(pre) == pre

    pre = 'word11-word**_word'
    post = 'word11 word** word'
    assert du(pre) == post


def test_spelling_preprocessor():
    sp = methods.spelling_preprocessor()

    pre = 'woard'
    post = 'board'
    assert sp(pre) == post

    pre = 'spelled spalled spellad speled spellet'
    post = 'spelled spelled spelled speed spelled'
    assert sp(pre) == post

    pre = 'jmup huose wrarior thraed wede chesnut watr'
    post = 'jump house warrior thread were chestnut water'
    assert sp(pre) == post

    pre = 'jump123 CaR777 angel**&&'
    assert sp(pre) == pre

    pre = 'jump     car     ostrich'
    post = 'jump car ostrich'
    assert sp(pre) == post

    pre = 'a can\'t do this'
    assert sp(pre) == pre

    pre = 'a caen\'t do this'
    post = 'a can\'t do this'
    assert sp(pre) == post

    # Test correction without pucntuation
    pre = 'a cantt do this'
    post = 'a cant do this'
    assert sp(pre) == post

def test_segment_preprocessor():
    sp = methods.segment_preprocessor()

    pre = ''
    assert sp(pre) == pre

    pre = 'cat'
    assert sp(pre) == pre

    pre = 'background doghouse lifetime elsewhere baseball basketball weatherman earthquake backbone'
    assert sp(pre) == pre

    pre = '  background   '
    post = 'background'
    assert sp(pre) == post

    # double space is on purpose to add error handling
    pre = 'groundback housedog timelife wherelse ballbase ballbasket manweather quakeearth boneback'
    post = 'ground back house dog time life where lse ball base ball basket man weather quake earth bone back'
    assert sp(pre) == post

    pre = 'andillnevergiveup'
    post = 'and ill never give up'
    assert sp(pre) == post

    pre = 'andillnevergiveupthebackgroundcheck'
    post = 'and ill never give up the background check'
    assert sp(pre) == post

    pre = 'AndI\'llnevergiveupthebackgroundcheck!'
    post = 'and ill never give up the background check'
    assert sp(pre) == post

    pre = 'backgrounddoghouselifetimeelsewherebaseballbasketballweathermanearthquakebackbone'
    post = 'background doghouse lifetime elsewhere baseball basketball weatherman earthquake backbone'
    assert sp(pre) == post

    pre = '#twitterhashtag'
    post = 'twitter hash tag'
    assert sp(pre) == post

    pre = '##$%##$genius##$$#blog$$#$@'
    post = 'genius blog'
    assert sp(pre) == post


def test_lowercase_preprocessor():
    lp = methods.lowercase_preprocessor()

    pre = 'and and and and  and cat cat cat cat'
    assert lp(pre) == pre
    assert lp('') == ''

    pre = '#$&**@#(*@(&%(@*#(@*!!!)__!@!)*@)!}{":?><|'
    assert lp(pre) == pre

    pre = 'OOldsoujqAhjfkjaAOWJIERkls'
    post = 'ooldsoujqahjfkjaaowjierkls'
    assert lp(pre) == post

    pre = '@#*$#$JJJJJJ@#*@(*#LLLLL*@#&*@#&LA DI D&&&'
    post = '@#*$#$jjjjjj@#*@(*#lllll*@#&*@#&la di d&&&'
    assert lp(pre) == post

def test_bpe_preprocessor():
    from preprocess import vocabulary
    import os
    import pickle

    corpus = 'notacorpus'
    bpe_method = 'bpe10'
    bpe_methods = [bpe_method]
    bpe_set = {'sm', 'a', 'll'}

    notacorpusdir = os.path.join(vocabulary.vocab_dir, corpus)
    if os.path.isdir(notacorpusdir):
        os.rmdir(notacorpusdir)

    os.mkdir(notacorpusdir)

    bpe_filename = vocabulary.create_bpe_set_filename(corpus, bpe_methods)

    if os.path.isfile(bpe_filename):
        os.remove(bpe_filename)

    with open(bpe_filename, 'wb') as f:
        pickle.dump(bpe_set, f)

    os.environ[methods.corpus_environment_name] = corpus
    os.environ[methods.methods_environment_name] = bpe_method

    bpep = methods.bpe_preprocessor()

    # Test basic functionality
    expected = 'sm a ll'
    input_word = 'small'
    assert bpep(input_word) == expected

    # Ensure that segments not in set will be properly reduced to single characters
    expected = 'a b e r n a t h y'
    input_word = 'abernathy'
    assert bpep(input_word) == expected

    os.remove(bpe_filename)
    os.rmdir(notacorpusdir)
    assert not os.path.isfile(bpe_filename)
    assert not os.path.isdir(notacorpusdir)


def test_create_preprocessor():

    from preprocess import methods

    tests = ['OOldsoujqAhjfkjaAOWJIERkls', 'groundback housedog timelife wherelse ballbase ballbasket manweather quakeearth boneback', 'spelled spalled spellad speled spellet', 'word_word_word_word-word-word-word', 'wording caresses flies dies mules denied died agreed owned humbled sized', '!@#$%^&*()_+}{|":?><~', 'w0rd   and w2134566rd and    %%%%%%##@@@@ 123556']

    preprocess_dict = {
                          'lc' : methods.lowercase_preprocessor(),
                          'ws' : methods.segment_preprocessor(),
                          'sc' : methods.spelling_preprocessor(),
                          'ud' : methods.dash_underscore_preprocessor(),
                          'st' : methods.stemming_preprocessor(),
                          'np' : methods.punctuation_preprocessor(),
                          'nr' : methods.number_preprocessor(),
                      }

    bp = methods.base_preprocessor()
    pp = methods.create_preprocessor([])

    # Test no methods
    for test in tests:
        assert bp(test) == pp(test)

    # Test correct order of segmentation and spelling correction
    s = 'huoseword'
    # Out of order methods should apply in reverse order
    treatments = ['sc', 'ws']
    expected = 'house word'
    pp = methods.create_preprocessor(treatments)
    assert pp(s) == expected

    # Test correct order of spelling correction and then stemming
    s = 'flyang'
    # correction should happen first even though its listed last
    treatments = ['st', 'sc']
    expected = 'fli'
    pp = methods.create_preprocessor(treatments)
    assert pp(s) == expected

    # Test noneffect of stopword removal
    pp = methods.create_preprocessor(['sp'])
    for test in tests:
        assert bp(test) == pp(test)

    # Test noneffect of rare word filtering
    pp = methods.create_preprocessor(['r5'])
    for test in tests:
        assert bp(test) == pp(test)

    # Test noneffect of hashing
    pp = methods.create_preprocessor(['h5'])
    for test in tests:
        assert bp(test) == pp(test)

    # Test noneffect of test method
    pp = methods.create_preprocessor(['tt'])
    for test in tests:
        assert bp(test) == pp(test)

    # Test noneffect of all methods with no effect together
    pp = methods.create_preprocessor(['r5', 'sp', 'h5'])
    for test in tests:
        assert bp(test) == pp(test)

def test_string_check():
    for name, val in methods.__dict__.items():
        if callable(val) and 'preprocessor' in name and name != 'create_preprocessor':
            preprocessor = val()

            try:
                preprocessor('something')
            except AttributeError:
                pytest.fail('Unexpected Attribute Error')

            with pytest.raises(AttributeError):
                preprocessor(1234)

            with pytest.raises(AttributeError):
                preprocessor(None)

def test_bpe_encode():
    bpee = methods.bpe_encode

    # Test basic functionality
    bpe_set = {'a', 'b', 'c', 'd'}
    expected = ['a', 'b', 'c', 'd']
    input_word = 'abcd'
    assert bpee(input_word, bpe_set) == expected

    expected = ['b', 'c', 'd', 'a']
    input_word = 'bcda'
    assert bpee(input_word, bpe_set) == expected

    # Test with word parts
    bpe_set = {'sm', 'a', 'll'}
    expected = ['sm', 'a', 'll']
    input_word = 'small'
    assert bpee(input_word, bpe_set) == expected

    # Test longer word
    expected = ['sm', 'a', 'll', 'e', 'r']
    input_word = 'smaller'
    assert bpee(input_word, bpe_set) == expected

    # Test longer word
    expected = ['a', 'b', 'e', 'r', 'n', 'a', 't', 'h', 'y']
    input_word = 'abernathy'
    assert bpee(input_word, bpe_set) == expected

    # test automatic partials
    input_word = 'smell'
    expected = ['sm', 'e', 'll']
    assert bpee(input_word, bpe_set) == expected

    bpe_set = {}
    input_word = 'smell'
    expected = ['s', 'm', 'e', 'l', 'l']
    assert bpee(input_word, bpe_set) == expected

    bpe_set = {'sm', 'al'}
    # Test non-usage of irrelevant words
    input_word = 'smee'
    expected = ['sm', 'e', 'e']
    assert bpee(input_word, bpe_set) == expected
