import pytest
from ..word_embedding_model import WordEmbeddingModel
from ..utils import load_weat_w2v
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText


def test_word_embedding_model_init_types():

    # Test types verifications

    # target sets None
    with pytest.raises(TypeError,
                       match='word_embedding must be an instance of a '
                       'gensim\'s KeyedVectors'):
        WordEmbeddingModel(None)

    # target sets int
    with pytest.raises(TypeError,
                       match='word_embedding must be an instance of a '
                       'gensim\'s KeyedVectors'):
        WordEmbeddingModel('abc')

    with pytest.raises(
            TypeError,
            match=
            'word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel(1)

    with pytest.raises(
            TypeError,
            match=
            'word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel({})

    with pytest.raises(
            TypeError,
            match=
            'word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel({})


def test_word_embedding_model_init():

    # Test

    # Load dummy w2v
    weat_we = load_weat_w2v()

    with pytest.raises(TypeError, match='model_name must be a string'):
        WordEmbeddingModel(weat_we, 12)

    with pytest.raises(TypeError,
                       match='vocab_prefix parameter must be a string.'):
        WordEmbeddingModel(weat_we, 'A', 12)

    model_1 = WordEmbeddingModel(weat_we, 'weat_we')
    assert model_1.model_ == weat_we
    assert model_1.model_name_ == 'weat_we'
    assert model_1.vocab_prefix_ == ''

    model_2 = WordEmbeddingModel(weat_we)
    assert model_2.model_name_ == 'Unnamed word embedding model'
    assert model_2.vocab_prefix_ == ''

    model_3 = WordEmbeddingModel(weat_we, 'weat_we', '\\c\\en')
    assert model_3.model_name_ == 'weat_we'
    assert model_3.vocab_prefix_ == '\\c\\en'


def test_word_embedding_model_eq():
    model_1 = WordEmbeddingModel(load_weat_w2v(), 'weat_1')
    model_2 = WordEmbeddingModel(load_weat_w2v(), 'weat_2')

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.model_ = None

    assert model_1 != model_2


def test_w2v():

    w2v = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1)
    w2v_keyed_vectors = w2v.wv
    wem = WordEmbeddingModel(w2v_keyed_vectors, "w2v")

    assert w2v.wv == wem.model_


def test_fast():
    fast = FastText(size=4,
                    window=3,
                    min_count=1,
                    sentences=common_texts,
                    iter=10)
    fast_keyed_vectors = fast.wv
    wem = WordEmbeddingModel(fast_keyed_vectors, "w2v")

    assert fast.wv == wem.model_
