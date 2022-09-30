"""Tests of the word embedding model module."""
import gensim
import numpy as np
import pytest
import semantic_version
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.test.utils import common_texts
from wefe.word_embedding_model import WordEmbeddingModel

gensim_version = semantic_version.Version.coerce(gensim.__version__)


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

    w2v = KeyedVectors.load("./wefe/datasets/data/test_model.kv")

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


def test__eq__(keyed_vector_model: gensim.models.KeyedVectors):
    model_1 = WordEmbeddingModel(keyed_vector_model, "w2v")
    model_2 = WordEmbeddingModel(keyed_vector_model, "w2v_2")
    model_3_prefix_a = WordEmbeddingModel(keyed_vector_model, "w2v_3", vocab_prefix="a")
    model_3_prefix_b = WordEmbeddingModel(keyed_vector_model, "w2v_3", vocab_prefix="b")

    assert model_1 != ""

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.wv = None

    assert model_1 != model_2

    assert model_2 != model_3_prefix_a
    assert model_3_prefix_a == model_3_prefix_a

    assert model_3_prefix_b != model_3_prefix_a


def test__contains__(model: WordEmbeddingModel):
    assert "men" in model
    assert "asdf" not in model
    assert None not in model
    assert 0 not in model


def test__getitem__(model: WordEmbeddingModel):

    embedding = model["ASDF"]
    assert embedding is None

    embedding = model["career"]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (300,)


def test__repr__(keyed_vector_model: gensim.models.KeyedVectors):

    model_1 = WordEmbeddingModel(keyed_vector_model, "w2v")
    model_1_no_name = WordEmbeddingModel(keyed_vector_model)
    model_1_prefix_a = WordEmbeddingModel(keyed_vector_model, "w2v", vocab_prefix="a")
    model_1_no_name_prefix_a = WordEmbeddingModel(keyed_vector_model, vocab_prefix="a")

    assert (
        model_1.__repr__()
        == "<WordEmbeddingModel named 'w2v' with 13013 word embeddings of 300 dims>"
    )
    assert model_1_no_name.__repr__() == (
        "<WordEmbeddingModel 'Unnamed model' with 13013 word embeddings of 300 dims>"
    )
    assert model_1_prefix_a.__repr__() == (
        "<WordEmbeddingModel named 'w2v' with 13013 word embeddings of 300 dims and"
        " 'a' as word prefix>"
    )
    assert model_1_no_name_prefix_a.__repr__() == (
        "<WordEmbeddingModel 'Unnamed model' with 13013 word embeddings of 300 dims "
        "and 'a' as word prefix>"
    )

    del model_1.name
    assert model_1.__repr__() == "<WordEmbeddingModel with wrong __repr__>"


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
def test_normalize_embeddings(model: WordEmbeddingModel):
    # test unnormalized embeddings
    for word in model.vocab:
        assert np.abs(np.linalg.norm(model[word]) - 1.0) > 0.000001

    # test normalized embeddings
    model.normalize()
    for word in model.vocab:
        assert np.linalg.norm(model[word]) - 1.0 < 0.000001

    model.wv = None

    with pytest.raises(
        TypeError, match="The model does not have the init_sims method implemented."
    ):
        model.normalize()


# -------------------------------------------------------------------------------------
def test_update_embedding(model: WordEmbeddingModel):

    new_embedding = np.ones(300, dtype=model.wv.vectors.dtype)
    model.update("The", new_embedding)
    assert model["The"].shape == (300,)
    assert all(model["The"] == new_embedding)

    with pytest.raises(TypeError, match=r"word should be a string, got .*"):
        model.update(0, new_embedding)

    with pytest.raises(ValueError, match=r"word .* not in model vocab."):
        model.update("blablablablabla", new_embedding)

    with pytest.raises(
        TypeError, match=r".* new embedding should be a np\.array, got .*\."
    ):
        model.update("The", 0)

    with pytest.raises(
        ValueError,
        match=r"The size of .* embedding (.*) is different from the size of the "
        r"embeddings in the model (.*)\.",
    ):
        model.update("The", np.ones(200, dtype=np.float64))

    with pytest.raises(
        ValueError, match=r"embedding dtype .* is not the same of model's dtype .*\."
    ):
        model.update("The", np.ones(300, dtype=np.float64))


# -------------------------------------------------------------------------------------
def test_update_embeddings(model):
    words = ["The", "in"]
    embeddings = [np.ones(300, dtype=np.float32), np.ones(300, dtype=np.float32) * -1]

    model.batch_update(words, embeddings)

    assert all(model["The"] == embeddings[0])
    assert all(model["in"] == embeddings[1])

    embeddings_in_array = np.array(
        [np.ones(300, dtype=np.float32) * 2, np.ones(300, dtype=np.float32) * -2]
    )
    model.batch_update(words, embeddings_in_array)

    assert all(model["The"] == embeddings_in_array[0])
    assert all(model["in"] == embeddings_in_array[1])

    with pytest.raises(
        TypeError,
        match=r"words argument should be a list, tuple or np.array of strings, got .*",
    ):
        model.batch_update(None, embeddings)

    with pytest.raises(
        TypeError,
        match=r"embeddings should be a list, tuple or np.array, got:.*",
    ):
        model.batch_update(words, None)

    with pytest.raises(
        ValueError,
        match=r"words and embeddings must have the same size, got:.*",
    ):
        model.batch_update(words + ["is"], embeddings)
