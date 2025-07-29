"""Unit tests for the WordEmbeddingModel class from wefe.word_embedding_model."""

import gensim
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.test.utils import common_texts
import numpy as np
import pytest

from wefe.word_embedding_model import GENSIM_V4_OR_GREATER, WordEmbeddingModel


@pytest.fixture
def test_keyed_vectors() -> KeyedVectors:
    """Fixture that loads a pre-trained KeyedVectors model for testing."""
    test_model = KeyedVectors.load("./wefe/datasets/data/test_model.kv")
    return test_model


def test__init__() -> None:
    """Test the constructor of the WordEmbeddingModel class.

    It verifies type validations for `wv`, `name`, and `vocab_prefix`
    arguments, and checks for correct initialization of model attributes.
    """
    # target sets None
    with pytest.raises(
        TypeError, match="wv must be an instance of gensim's BaseKeyedVectors, but got"
    ):
        WordEmbeddingModel(wv=None)

    # target sets int
    with pytest.raises(
        TypeError,
        match=(
            "wv must be an instance of gensim's BaseKeyedVectors, but got"
            " <class 'str'>."
        ),
    ):
        WordEmbeddingModel(wv="abc")

    with pytest.raises(
        TypeError,
        match=(
            "wv must be an instance of gensim's BaseKeyedVectors, but got"
            " <class 'int'>."
        ),
    ):
        WordEmbeddingModel(wv=1)

    with pytest.raises(
        TypeError,
        match=(
            "wv must be an instance of gensim's BaseKeyedVectors, but got "
            "<class 'dict'>."
        ),
    ):
        WordEmbeddingModel(wv={})

    w2v = KeyedVectors.load("./wefe/datasets/data/test_model.kv")

    with pytest.raises(
        TypeError,
        match="name must be a string or None, but got <class 'int'>.",
    ):
        WordEmbeddingModel(wv=w2v, name=1)

    with pytest.raises(
        TypeError,
        match=r"vocab_prefix must be a string or None, but got <class 'int'>.",
    ):
        WordEmbeddingModel(wv=w2v, vocab_prefix=1)

    # test models
    model = WordEmbeddingModel(wv=w2v)
    assert model.wv == w2v
    assert model.name == "Unnamed model"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(wv=w2v, name="w2v_sm")
    assert model.wv == w2v
    assert model.name == "w2v_sm"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(wv=w2v, name="w2v_sm", vocab_prefix="\\c\\en")
    assert model.wv == w2v
    assert model.name == "w2v_sm"
    assert model.vocab_prefix == "\\c\\en"


def test__eq__(test_keyed_vectors: gensim.models.KeyedVectors) -> None:
    """Test the equality comparison (__eq__) operator for WordEmbeddingModel.

    It checks if two WordEmbeddingModel instances are equal based on their internal
    KeyedVectors object, name, and vocab_prefix.
    """
    model_1 = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")
    model_2 = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v_2")
    model_3_prefix_a = WordEmbeddingModel(
        wv=test_keyed_vectors, name="w2v_3", vocab_prefix="a"
    )
    model_3_prefix_b = WordEmbeddingModel(
        wv=test_keyed_vectors, name="w2v_3", vocab_prefix="b"
    )

    assert model_1 != ""

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.wv = None

    assert model_1 != model_2

    assert model_2 != model_3_prefix_a
    assert model_3_prefix_a == model_3_prefix_a

    assert model_3_prefix_b != model_3_prefix_a


def test__contains__(test_keyed_vectors: gensim.models.KeyedVectors) -> None:
    """Test the __contains__ operator for WordEmbeddingModel.

    It verifies if a word exists in the model's vocabulary.
    """
    model = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")

    assert "men" in model
    assert "asdf" not in model
    assert None not in model
    assert 0 not in model


def test__getitem__(test_keyed_vectors: gensim.models.KeyedVectors) -> None:
    """Test the __getitem__ operator for WordEmbeddingModel.

    It checks if a word embedding can be retrieved correctly
    and raises KeyError for words not in the vocabulary.
    """
    model = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")

    embedding = model["career"]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (300,)

    with pytest.raises(KeyError, match=r"word 'not_in_vocab' not in model vocab."):
        model["not_in_vocab"]


def test__repr__(test_keyed_vectors: gensim.models.KeyedVectors) -> None:
    """Test the string representation (__repr__) of WordEmbeddingModel.

    It verifies that the representation correctly displays model name,
    vocabulary size, embedding dimensions, and vocab prefix when present.
    """
    model_1 = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")
    model_1_no_name = WordEmbeddingModel(wv=test_keyed_vectors)
    model_1_prefix_a = WordEmbeddingModel(
        wv=test_keyed_vectors, name="w2v", vocab_prefix="a"
    )
    model_1_no_name_prefix_a = WordEmbeddingModel(
        wv=test_keyed_vectors, vocab_prefix="a"
    )

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


def test__init__with_w2v_model() -> None:
    """Test initialization of WordEmbeddingModel with a gensim.Word2Vec model.

    Ensures that the internal wv attribute correctly points to the Word2Vec's
    KeyedVectors.
    """
    if GENSIM_V4_OR_GREATER:
        w2v = Word2Vec(
            sentences=common_texts, vector_size=100, window=5, min_count=1, workers=-1
        )
    else:
        w2v = Word2Vec(
            sentences=common_texts, size=100, window=5, min_count=1, workers=-1
        )
    w2v_keyed_vectors = w2v.wv
    wem = WordEmbeddingModel(wv=w2v_keyed_vectors, name="w2v")

    assert w2v.wv == wem.wv


def test__init_with_fast_model() -> None:
    """Test initialization of WordEmbeddingModel with a gensim.FastText model.

    Ensures that the internal wv attribute correctly points to the FastText's
    KeyedVectors.
    """
    if GENSIM_V4_OR_GREATER:
        fast = FastText(
            vector_size=4, window=3, min_count=1, sentences=common_texts, epochs=10
        )
    else:
        fast = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)
    fast_keyed_vectors = fast.wv
    wem = WordEmbeddingModel(wv=fast_keyed_vectors, name="w2v")

    assert fast.wv == wem.wv


def test_normalize_embeddings(test_keyed_vectors: gensim.models.KeyedVectors) -> None:
    """Test the normalize method of WordEmbeddingModel.

    It checks if embeddings are correctly normalized to unit length and
    handles cases where the internal gensim model cannot be normalized.
    """
    model = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")

    # test unnormalized embeddings
    for word in model.vocab:
        assert np.abs(np.linalg.norm(model[word]) - 1.0) > 0.000001

    # test normalized embeddings
    model.normalize()
    for word in model.vocab:
        assert np.linalg.norm(model[word]) - 1.0 < 0.000001

    model.wv = None

    with pytest.raises(
        AttributeError,
        match=(
            r"The underlying gensim model does not have a known normalization method "
            r"\('get_normed_vectors' or 'init_sims'\)."
        ),
    ):
        model.normalize()


def test_update_embeddings(
    test_keyed_vectors: gensim.models.KeyedVectors,
) -> None:
    """Test the batch_update method of WordEmbeddingModel.

    It checks if multiple word embeddings can be updated simultaneously
    and validates input types and sizes.
    """
    model = WordEmbeddingModel(wv=test_keyed_vectors, name="w2v")

    words = ["The", "in"]
    embeddings = [
        np.ones(300, dtype=np.float32),
        np.ones(300, dtype=np.float32) * -1,
    ]

    model.batch_update(words=words, embeddings=embeddings)

    assert all(model["The"] == embeddings[0])
    assert all(model["in"] == embeddings[1])

    embeddings_in_array = np.array(
        [
            np.ones(300, dtype=np.float32) * 2,
            np.ones(300, dtype=np.float32) * -2,
        ]
    )
    model.batch_update(words=words, embeddings=embeddings_in_array)

    assert all(model["The"] == embeddings_in_array[0])
    assert all(model["in"] == embeddings_in_array[1])

    # words argument should be a list, tuple or np.array of strings
    with pytest.raises(
        TypeError,
        match=(
            r"words argument should be a list, tuple or np.array of strings, "
            r"but got <class 'NoneType'>."
        ),
    ):
        model.batch_update(words=None, embeddings=embeddings)

    # embeddings argument should be a list, tuple or np.array of NumPy arrays
    with pytest.raises(
        TypeError,
        match=(
            r"embeddings argument should be a list, tuple or np.array of NumPy arrays, "
            r"but got <class 'NoneType'>."
        ),
    ):
        model.batch_update(words=words, embeddings=None)

    # words and embeddings must have the same size
    with pytest.raises(
        ValueError,
        match=(
            r"words and embeddings must have the same number of elements, "
            "but got 3 words and 2 embeddings."
        ),
    ):
        model.batch_update(words=words + ["is"], embeddings=embeddings)

    # Check for non-string word in words
    with pytest.raises(
        TypeError,
        match=r"All elements in 'words' must be strings, but found a <class 'int'>.",
    ):
        model.batch_update(words=["The", 1], embeddings=embeddings)

    # Check for word not in vocab
    with pytest.raises(
        ValueError,
        match=r"The following words are not in the model's vocabulary: not_in_vocab.",
    ):
        model.batch_update(words=["The", "not_in_vocab"], embeddings=embeddings)

    # Check for non-numpy embedding in embeddings
    with pytest.raises(
        TypeError,
        match=(
            r"Embedding at index 1 \('in'\) is not a NumPy array, but "
            r"got <class 'int'>."
        ),
    ):
        model.batch_update(words=words, embeddings=[embeddings[0], 1])

    # Check for wrong embedding shape
    with pytest.raises(
        ValueError,
        match=(
            r"Embedding at index 1 \('in'\) has shape \(200,\) which is different from "
            r"the model's embedding size \(300,\)."
        ),
    ):
        model.batch_update(
            words=words,
            embeddings=[embeddings[0], np.ones(200, dtype=np.float32)],
        )

    # Check for wrong embedding dtype
    with pytest.raises(
        ValueError,
        match=(
            r"Embedding at index 1 \('in'\) with dtype \(float64\) cannot be safely "
            r"cast to model's dtype \(float32\)."
        ),
    ):
        model.batch_update(
            words=words,
            embeddings=[embeddings[0], np.ones(300, dtype=np.float64)],
        )
