import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors

from wefe.word_embedding_model import WordEmbeddingModel
from wefe.datasets.datasets import load_weat
from wefe.preprocessing import get_embeddings_from_word_set, preprocess_word


@pytest.fixture
def model():
    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")
    return WordEmbeddingModel(w2v, "word2vec")


@pytest.fixture
def weat_word_set():
    return load_weat()


def test_preprocess_word():

    word = preprocess_word("Woman")
    assert word == "Woman"

    word = preprocess_word("Woman", {"lowercase": True})
    assert word == "woman"

    word = preprocess_word("Woman", {"uppercase": True})
    assert word == "WOMAN"

    word = preprocess_word("Woman", {"titlecase": True})
    assert word == "Woman"

    word = preprocess_word("Woman", {"lowercase": True, "case": True, "title": True})
    assert word == "woman"

    word = preprocess_word("wömàn", {"strip_accents": True})
    assert word == "woman"

    word = preprocess_word("wömàn", {"strip_accents": "ascii"})
    assert word == "woman"

    word = preprocess_word("wömàn", {"strip_accents": "unicode"})
    assert word == "woman"

    # all together
    word = preprocess_word("WöMàn", {"lowercase": True, "strip_accents": True})
    assert word == "woman"

    word = preprocess_word("WöMàn", {"uppercase": True, "strip_accents": True})
    assert word == "WOMAN"

    word = preprocess_word("WöMàn", {"titlecase": True, "strip_accents": True})
    assert word == "Woman"

    # check for custom preprocessor
    word = preprocess_word("Woman", {"preprocessor": lambda x: x.lower()})
    assert word == "woman"

    # check if preprocessor overrides any other option
    word = preprocess_word(
        "Woman", {"preprocessor": lambda x: x.upper(), "lowercase": True}
    )
    assert word == "WOMAN"

    # now with prefix
    word = preprocess_word("woman", vocab_prefix="asd-")
    assert word == "asd-woman"


def test_get_embeddings_from_word_set_types(model):

    WORDS = ["man", "woman"]

    with pytest.raises(
        TypeError, match=r"model should be a WordEmbeddingModel instance, got .*\."
    ):
        get_embeddings_from_word_set(None, None)

    with pytest.raises(
        TypeError,
        match=r"word_set should be a list, tuple or np.array of strings, got.*",
    ):
        get_embeddings_from_word_set(model, word_set=None)

    with pytest.raises(
        TypeError,
        match=r"preprocessors should be a list of dicts which contains preprocessor options, got .*\.",
    ):
        get_embeddings_from_word_set(model, WORDS, preprocessors=None)

    with pytest.raises(
        TypeError,
        match=(
            r"preprocessors must indicate at least one preprocessor, even if it is "
            r"an empty dictionary {}, got: .*\."
        ),
    ):
        get_embeddings_from_word_set(model, WORDS, preprocessors=[])

    with pytest.raises(
        TypeError, match=r"each preprocessor should be a dict, got .* at index .*\."
    ):
        get_embeddings_from_word_set(
            model, WORDS, preprocessors=[{"lower": True}, {"upper": True}, 1]
        )

    with pytest.raises(
        ValueError, match=r"strategy should be 'first' or 'all', got .*\."
    ):
        get_embeddings_from_word_set(model, WORDS, strategy=None)

    with pytest.raises(
        ValueError, match=r"strategy should be 'first' or 'all', got .*\."
    ):
        get_embeddings_from_word_set(model, WORDS, strategy="blabla")


def test_get_embeddings_from_word_set(model):

    # ----------------------------------------------------------------------------------
    # test basic operation of _get_embeddings_from_word_set
    WORDS = ["man", "woman"]

    not_found_words, embeddings = get_embeddings_from_word_set(model, WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test with a word that does not exists in the model
    WORDS = ["man", "woman", "not_a_word_"]
    not_found_words, embeddings = get_embeddings_from_word_set(model, WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 1

    assert list(embeddings.keys()) == ["man", "woman"]
    assert ["not_a_word_"] == not_found_words

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test word preprocessor lowercase
    WORDS = [
        "mAN",
        "WOmaN",
    ]
    not_found_words, embeddings = get_embeddings_from_word_set(
        model, WORDS, [{"lowercase": True}]
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test word preprocessor strip_accents:
    WORDS = [
        "mán",
        "wömàn",
    ]
    not_found_words, embeddings = get_embeddings_from_word_set(
        model, WORDS, [{"strip_accents": True}]
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test two word preprocessors strip_accents strategy="first":
    WORDS = [
        "mán",
        "WöMàn",
    ]
    not_found_words, embeddings = get_embeddings_from_word_set(
        model,
        WORDS,
        [
            {"strip_accents": True},
            {"strip_accents": True, "lowercase": True},
            {"strip_accents": True, "uppercase": True},
        ],
        strategy="first",
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 1

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == ["WöMàn"]

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test two word preprocessors strip_accents strategy="all":
    WORDS = [
        "mán",
        "WöMàn",
    ]
    not_found_words, embeddings = get_embeddings_from_word_set(
        model,
        WORDS,
        [
            {"strip_accents": True},
            {"strip_accents": True, "lowercase": True},
            {"strip_accents": True, "uppercase": True},
            {"strip_accents": True, "titlecase": True},
        ],
        strategy="all",
    )

    assert list(embeddings.keys()) == ["man", "MAN", "Man", "woman", "WOMAN", "Woman"]
    assert not_found_words == ["WöMàn"]

    assert [np.array_equal(model[k], embeddings[k]) for k in embeddings.keys()]

    # ----------------------------------------------------------------------------------
    # test normalize
    WORDS = ["man", "woman"]

    _, embeddings = get_embeddings_from_word_set(model, WORDS, normalize=True)

    assert 0.99999 < np.linalg.norm(embeddings["man"]) < 1.00001
    assert 0.99999 < np.linalg.norm(embeddings["woman"]) < 1.00001
