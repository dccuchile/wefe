import gensim
import pytest
import numpy as np
import logging

import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText
import semantic_version

from ..query import Query
from ..datasets import load_weat
from ..word_embedding_model import WordEmbeddingModel
from ..utils import load_weat_w2v

LOGGER = logging.getLogger(__name__)
gensim_version = semantic_version.Version.coerce(gensim.__version__)


def test_word_embedding_init_types():

    # Test types verifications

    # target sets None
    with pytest.raises(TypeError,
                       match="model should be an instance of gensim's BaseKeyedVectors"):
        WordEmbeddingModel(None)

    # target sets int
    with pytest.raises(TypeError,
                       match="model should be an instance of gensim's BaseKeyedVectors"):
        WordEmbeddingModel('abc')

    with pytest.raises(TypeError,
                       match="model should be an instance of gensim's BaseKeyedVectors"):
        WordEmbeddingModel(1)

    with pytest.raises(TypeError,
                       match="model should be an instance of gensim's BaseKeyedVectors"):
        WordEmbeddingModel({})


def test_word_embedding_init():

    # Load dummy w2v
    weat_we = load_weat_w2v()

    # test models
    model = WordEmbeddingModel(weat_we)
    assert model.model == weat_we
    assert model.model_name == 'Unnamed word embedding model'
    assert model.vocab_prefix == None

    model = WordEmbeddingModel(weat_we, 'weat_we')
    assert model.model == weat_we
    assert model.model_name == 'weat_we'
    assert model.vocab_prefix == None

    model = WordEmbeddingModel(weat_we, 'weat_we', '\\c\\en')
    assert model.model == weat_we
    assert model.model_name == 'weat_we'
    assert model.vocab_prefix == '\\c\\en'


def test_word_embedding_eq():
    model_1 = WordEmbeddingModel(load_weat_w2v(), 'weat_1')
    model_2 = WordEmbeddingModel(load_weat_w2v(), 'weat_2')
    model_3 = WordEmbeddingModel(load_weat_w2v(), 'weat_2', vocab_prefix='a')

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.model = None

    assert model_1 != model_2

    assert model_2 != model_3
    assert model_3 == model_3


def test_w2v():

    if gensim_version.major >= 4:
        w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=-1)
    else:
        w2v = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1)
    w2v_keyed_vectors = w2v.wv
    wem = WordEmbeddingModel(w2v_keyed_vectors, "w2v")

    assert w2v.wv == wem.model


def test_fast():

    if gensim_version.major >= 4:
        fast = FastText(vector_size=4,
                        window=3,
                        min_count=1,
                        sentences=common_texts,
                        epochs=10)
    else:
        fast = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)
    fast_keyed_vectors = fast.wv
    wem = WordEmbeddingModel(fast_keyed_vectors, "w2v")

    assert fast.wv == wem.model


def test__getitem__():
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v')

    embedding = model['ASDF']
    assert embedding == None

    embedding = model['career']
    assert isinstance(embedding, np.ndarray)


def test_preprocess_word():

    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    word = model._preprocess_word('Woman')
    assert word == 'Woman'

    word = model._preprocess_word('Woman', {'lowercase': True})
    assert word == 'woman'

    word = model._preprocess_word('wömàn', {'strip_accents': True})
    assert word == 'woman'

    word = model._preprocess_word('wömàn', {'strip_accents': 'ascii'})
    assert word == 'woman'

    word = model._preprocess_word('wömàn', {'strip_accents': 'unicode'})
    assert word == 'woman'

    # all together
    word = model._preprocess_word('WöMàn', {
        'lowercase': True,
        'strip_accents': True,
    })
    assert word == 'woman'

    # check for custom preprocessor
    word = model._preprocess_word('Woman', {'preprocessor': lambda x: x.lower()})
    assert word == 'woman'

    # check if preprocessor overrides any other option
    word = model._preprocess_word('Woman', {
        'preprocessor': lambda x: x.upper(),
        'lowercase': True
    })
    assert word == 'WOMAN'

    # now with prefix
    model = WordEmbeddingModel(w2v, 'weat_w2v', 'asd-')
    word = model._preprocess_word('woman')
    assert word == 'asd-woman'


def test_get_embeddings_from_word_set():

    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')
    WORDS = ['man', 'woman']

    with pytest.raises(TypeError, match="word_set should be a list of strings, got"):
        model.get_embeddings_from_word_set(None, preprocessor_args=1)

    with pytest.raises(
            TypeError,
            match="preprocessor_args should be a dict of preprocessor arguments, got"):
        model.get_embeddings_from_word_set(WORDS, preprocessor_args=1)

    with pytest.raises(
            TypeError,
            match=
            "secondary_preprocessor_args should be a dict of preprocessor arguments or "
            "None, got"):
        model.get_embeddings_from_word_set(WORDS, secondary_preprocessor_args=-1)

    # ----------------------------------------------------------------------------------
    # test basic opretaion of _get_embeddings_from_word_set
    WORDS = ['man', 'woman']

    not_found_words, embeddings = model.get_embeddings_from_word_set(WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ['man', 'woman']
    assert not_found_words == []

    assert np.array_equal(w2v['man'], embeddings['man'])
    assert np.array_equal(w2v['woman'], embeddings['woman'])

    # test with a word that does not exists in the model
    WORDS = ['man', 'woman', 'pizza']
    not_found_words, embeddings = model.get_embeddings_from_word_set(WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 1

    assert list(embeddings.keys()) == ['man', 'woman']
    assert ['pizza'] == not_found_words

    assert np.array_equal(w2v['man'], embeddings['man'])
    assert np.array_equal(w2v['woman'], embeddings['woman'])

    # ----------------------------------------------------------------------------------
    # test word preprocessor lowercase
    WORDS = [
        'mAN',
        'WOmaN',
    ]
    not_found_words, embeddings = model.get_embeddings_from_word_set(
        WORDS, {'lowercase': True})

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ['man', 'woman']
    assert not_found_words == []

    assert np.array_equal(w2v['man'], embeddings['man'])
    assert np.array_equal(w2v['woman'], embeddings['woman'])

    # ----------------------------------------------------------------------------------
    # test word preprocessor strip_accents:
    WORDS = [
        'mán',
        'wömàn',
    ]
    not_found_words, embeddings = model.get_embeddings_from_word_set(
        WORDS, {'strip_accents': True})

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ['man', 'woman']
    assert not_found_words == []

    assert np.array_equal(w2v['man'], embeddings['man'])
    assert np.array_equal(w2v['woman'], embeddings['woman'])

    # ----------------------------------------------------------------------------------
    # secondary_preprocessor_options:
    WORDS = ['mán', 'wömàn', 'qwerty', 'ásdf']
    not_found_words, embeddings = model.get_embeddings_from_word_set(
        WORDS, secondary_preprocessor_args={'strip_accents': True})

    assert len(embeddings) == 2
    assert len(not_found_words) == 2

    assert list(embeddings.keys()) == ['man', 'woman']
    assert not_found_words == ['qwerty', 'ásdf']

    assert np.array_equal(w2v['man'], embeddings['man'])
    assert np.array_equal(w2v['woman'], embeddings['woman'])


def test_warn_not_found_words(caplog):
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    model._warn_not_found_words('Set1', ['aaa', 'bbb'])
    assert "The following words from set 'Set1' do not exist within the vocabulary" in caplog.text
    assert "['aaa', 'bbb']" in caplog.text


@pytest.fixture
def simple_model_and_query():
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')
    weat_wordsets = load_weat()

    flowers = weat_wordsets['flowers']
    insects = weat_wordsets['insects']
    pleasant = weat_wordsets['pleasant_5']
    unpleasant = weat_wordsets['unpleasant_5']
    query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    return w2v, model, query, flowers, insects, pleasant, unpleasant


def test_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # test types

    # target sets None
    with pytest.raises(TypeError, match="query should be an instance of Query, got"):
        model.get_embeddings_from_query(None)

    # target sets int
    with pytest.raises(
            TypeError,
            match="lost_vocabulary_threshold should be float or np.floating, got"):
        model.get_embeddings_from_query(query, lost_vocabulary_threshold=None)

    with pytest.raises(
            TypeError,
            match="preprocessor_args should be a dict of preprocessor arguments, got"):
        model.get_embeddings_from_query(query, preprocessor_args=1)

    with pytest.raises(
            TypeError,
            match=
            "secondary_preprocessor_args should be a dict of preprocessor arguments or None, got"
    ):
        model.get_embeddings_from_query(query, secondary_preprocessor_args=1)

    with pytest.raises(TypeError, match="warn_not_found_words should be a boolean, got"):
        model.get_embeddings_from_query(query, warn_not_found_words=None)

    embeddings = model.get_embeddings_from_query(query)

    target_embeddings, attribute_embeddings = embeddings

    target_embeddings_sets = list(target_embeddings.values())
    attribute_embeddings_sets = list(attribute_embeddings.values())

    target_embeddings_names = list(target_embeddings.keys())
    attribute_embeddings_names = list(attribute_embeddings.keys())

    assert len(target_embeddings_sets) == 2
    assert len(attribute_embeddings_sets) == 2
    assert len(target_embeddings_names) == 2
    assert len(attribute_embeddings_names) == 2

    # test set names
    assert target_embeddings_names[0] == 'Flowers'
    assert target_embeddings_names[1] == 'Insects'
    assert attribute_embeddings_names[0] == 'Pleasant'
    assert attribute_embeddings_names[1] == 'Unpleasant'

    # test set embeddings
    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != 'axe', insects))
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]['aster'] == w2v['aster'])
    assert list(target_embeddings_sets[1]['ant'] == w2v['ant'])
    assert list(attribute_embeddings_sets[0]['caress'] == w2v['caress'])
    assert list(attribute_embeddings_sets[1]['abuse'] == w2v['abuse'])


def test_preprocessor_param_on_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # with lost words and warn_not_found_words=True
    flowers_2 = flowers + ['aaa', 'bbb']
    query_2 = Query([flowers_2, insects], [pleasant, unpleasant], ['Flowers', 'Insects'],
                    ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query_2, warn_not_found_words=True)
    assert "The following words from set 'Flowers' do not exist within the vocabulary" in caplog.text
    assert "['aaa', 'bbb']" in caplog.text

    # with preprocessor options
    flowers_3 = [s.upper() for s in flowers]
    query_3 = Query([flowers_3, insects], [pleasant, unpleasant], ['Flowers', 'Insects'],
                    ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query_3,
                                                 preprocessor_args={'lowercase': True})

    target_embeddings, attribute_embeddings = embeddings

    target_embeddings_sets = list(target_embeddings.values())
    attribute_embeddings_sets = list(attribute_embeddings.values())

    assert len(target_embeddings_sets) == 2
    assert len(attribute_embeddings_sets) == 2

    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != 'axe', insects))
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]['aster'] == w2v['aster'])
    assert list(target_embeddings_sets[1]['ant'] == w2v['ant'])
    assert list(attribute_embeddings_sets[0]['caress'] == w2v['caress'])
    assert list(attribute_embeddings_sets[1]['abuse'] == w2v['abuse'])

    # with secondary_preprocessor_options options
    embeddings = model.get_embeddings_from_query(
        query_3, secondary_preprocessor_args={'lowercase': True})

    target_embeddings, attribute_embeddings = embeddings

    target_embeddings_sets = list(target_embeddings.values())
    attribute_embeddings_sets = list(attribute_embeddings.values())

    assert len(target_embeddings_sets) == 2
    assert len(attribute_embeddings_sets) == 2

    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != 'axe', insects))
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]['aster'] == w2v['aster'])
    assert list(target_embeddings_sets[1]['ant'] == w2v['ant'])
    assert list(attribute_embeddings_sets[0]['caress'] == w2v['caress'])
    assert list(attribute_embeddings_sets[1]['abuse'] == w2v['abuse'])


def test_threshold_param_on_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # with lost vocabulary theshold.
    flowers_ = flowers + ['aaa', 'aab', 'aac', 'aad', 'aaf', 'aag', 'aah', 'aai', 'aaj']
    query = Query([flowers_, insects], [pleasant, unpleasant], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Flowers' into" in caplog.text

    # with lost vocabulary theshold.
    insects_ = insects + ['aaa', 'aab', 'aac', 'aad', 'aaf', 'aag', 'aah', 'aai', 'aaj']
    query = Query([flowers, insects_], [pleasant, unpleasant], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Insects' into" in caplog.text

    # with lost vocabulary theshold.
    pleasant_ = pleasant + [
        'aaa', 'aab', 'aac', 'aad', 'aaf', 'aag', 'aah', 'aai', 'aaj'
    ]
    query = Query([flowers, insects], [pleasant_, unpleasant], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Pleasant' into" in caplog.text

    # test attribute 2 with lost vocabulary theshold.
    unpleasant_ = insects + [
        'aaa', 'aab', 'aac', 'aad', 'aaf', 'aag', 'aah', 'aai', 'aaj'
    ]
    query = Query([flowers, insects], [pleasant, unpleasant_], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Unpleasant' into" in caplog.text

    # with lost vocabulary theshold.
    unpleasant_ = insects + [
        'aaa', 'aab', 'aac', 'aad', 'aaf', 'aag', 'aah', 'aai', 'aaj'
    ]
    query = Query([flowers, insects], [pleasant, unpleasant_], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.5)

    assert embeddings is not None
