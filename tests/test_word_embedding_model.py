"Tests of the word embedding model module"
import gensim
import pytest
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText, KeyedVectors
import semantic_version

from wefe.word_embedding_model import WordEmbeddingModel

gensim_version = semantic_version.Version.coerce(gensim.__version__)


@pytest.fixture
def word2vec_test():
    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")
    return WordEmbeddingModel(w2v, "word2vec")


def test__init__():

    # Test types verifications

    # target sets None
    with pytest.raises(
        TypeError, match="wv should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel(None)

    # target sets int
    with pytest.raises(
        TypeError, match="wv should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel("abc")

    with pytest.raises(
        TypeError, match="wv should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel(1)

    with pytest.raises(
        TypeError, match="wv should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel({})

    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")

    with pytest.raises(TypeError, match=r"name should be a string or None, got"):
        WordEmbeddingModel(w2v, name=1)

    with pytest.raises(
        TypeError, match=r"vocab_prefix should be a string or None, got"
    ):
        WordEmbeddingModel(w2v, vocab_prefix=1)

    # test models
    model = WordEmbeddingModel(w2v)
    assert model.wv == w2v
    assert model.name == "Unnamed model"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(w2v, "w2v_sm")
    assert model.wv == w2v
    assert model.name == "w2v_sm"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(w2v, "w2v_sm", "\\c\\en")
    assert model.wv == w2v
    assert model.name == "w2v_sm"
    assert model.vocab_prefix == "\\c\\en"


def test__eq__(word2vec_test):
    model_1 = WordEmbeddingModel(word2vec_test.wv, "w2v")
    model_2 = WordEmbeddingModel(word2vec_test.wv, "w2v_2")
    model_3 = WordEmbeddingModel(word2vec_test.wv, "w2v_3", vocab_prefix="a")
    model_3_ = WordEmbeddingModel(word2vec_test.wv, "w2v_3", vocab_prefix="b")

    assert model_1 != ""

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.wv = None

    assert model_1 != model_2

    assert model_2 != model_3
    assert model_3 == model_3

    assert model_3_ != model_3


def test__contains__(word2vec_test):
    assert "men" in word2vec_test
    assert "asdf" not in word2vec_test


def test__getitem__(word2vec_test):

    embedding = word2vec_test["ASDF"]
    assert embedding is None

    embedding = word2vec_test["career"]
    assert isinstance(embedding, np.ndarray)


def test__init__with_w2v_model():

    if gensim_version.major >= 4:
        w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=-1)
    else:
        w2v = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1)
    w2v_keyed_vectors = w2v.wv
    wem = WordEmbeddingModel(w2v_keyed_vectors, "w2v")

    assert w2v.wv == wem.wv


def test__init_with_fast_model():

    if gensim_version.major >= 4:
        fast = FastText(
            vector_size=4, window=3, min_count=1, sentences=common_texts, epochs=10
        )
    else:
        fast = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)
    fast_keyed_vectors = fast.wv
    wem = WordEmbeddingModel(fast_keyed_vectors, "w2v")

    assert fast.wv == wem.wv


# -------------------------------------------------------------------------------------
def test_normalize_embeddings(word2vec_test):
    # test unnormalized embeddings
    for word in word2vec_test.vocab:
        assert np.abs(np.linalg.norm(word2vec_test[word]) - 1.0) > 0.000001

    # test normalized embeddings
    word2vec_test.normalize()
    for word in word2vec_test.vocab:
        assert np.linalg.norm(word2vec_test[word]) - 1.0 < 0.000001

    word2vec_test.wv = None

    with pytest.raises(
        TypeError, match="The model does not have the init_sims method implemented."
    ):
        word2vec_test.normalize()


# -------------------------------------------------------------------------------------
def test_update_embedding(word2vec_test):
    new_embedding = np.ones(300, dtype=word2vec_test.wv.vectors.dtype)
    word2vec_test.update("The", new_embedding)
    assert all(word2vec_test["The"] == new_embedding)

    with pytest.raises(TypeError, match=r"word should be a string, got .*"):
        word2vec_test.update(0, new_embedding)

    with pytest.raises(ValueError, match=r"word .* not in model vocab."):
        word2vec_test.update("blablablablabla", new_embedding)

    with pytest.raises(
        TypeError, match=r".* new embedding should be a np\.array, got .*\."
    ):
        word2vec_test.update("The", 0)

    with pytest.raises(
        ValueError,
        match=r"The size of .* embedding (.*) is different from the size of the "
        r"embeddings in the model (.*)\.",
    ):
        word2vec_test.update("The", np.ones(200, dtype=np.float64))

    with pytest.raises(
        ValueError, match=r"embedding dtype .* is not the same of model's dtype .*\."
    ):
        word2vec_test.update("The", np.ones(300, dtype=np.float64))


# -------------------------------------------------------------------------------------
def test_update_embeddings(word2vec_test):
    words = ["The", "in"]
    embeddings = [np.ones(300, dtype=np.float32), np.ones(300, dtype=np.float32) * -1]

    word2vec_test.batch_update(words, embeddings)

    assert all(word2vec_test["The"] == embeddings[0])
    assert all(word2vec_test["in"] == embeddings[1])

    embeddings_in_array = np.array(
        [np.ones(300, dtype=np.float32) * 2, np.ones(300, dtype=np.float32) * -2]
    )
    word2vec_test.batch_update(words, embeddings_in_array)

    assert all(word2vec_test["The"] == embeddings_in_array[0])
    assert all(word2vec_test["in"] == embeddings_in_array[1])

    with pytest.raises(
        TypeError,
        match=r"words argument should be a list, tuple or np.array of strings, got .*",
    ):
        word2vec_test.batch_update(None, embeddings)

    with pytest.raises(
        TypeError, match=r"embeddings should be a list, tuple or np.array, got:.*",
    ):
        word2vec_test.batch_update(words, None)

    with pytest.raises(
        ValueError, match=r"words and embeddings must have the same size, got:.*",
    ):
        word2vec_test.batch_update(words + ["is"], embeddings)
