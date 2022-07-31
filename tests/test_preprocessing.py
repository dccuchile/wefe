import logging

import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors
from wefe.datasets.datasets import load_weat
from wefe.preprocessing import (
    _warn_not_found_words,
    get_embeddings_from_query,
    get_embeddings_from_set,
    get_embeddings_from_sets,
    preprocess_word,
)
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


@pytest.fixture
def model() -> WordEmbeddingModel:
    """Load a subset of Word2vec as a testing model.

    Returns
    -------
    WordEmbeddingModel
        The loaded testing model.
    """
    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")
    return WordEmbeddingModel(w2v, "word2vec")


@pytest.fixture
def weat_word_set():
    return load_weat()


@pytest.fixture
def simple_query():
    weat_wordsets = load_weat()

    flowers = weat_wordsets["flowers"]
    insects = weat_wordsets["insects"]
    pleasant = weat_wordsets["pleasant_5"]
    unpleasant = weat_wordsets["unpleasant_5"]
    query = Query(
        [flowers, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    return query, flowers, insects, pleasant, unpleasant


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
        get_embeddings_from_set(None, None)

    with pytest.raises(
        TypeError,
        match=r"word_set should be a list, tuple or np.array of strings, got.*",
    ):
        get_embeddings_from_set(model, word_set=None)

    with pytest.raises(
        TypeError,
        match=(
            r"preprocessors should be a list of dicts which contains preprocessor "
            r"options, got .*\."
        ),
    ):
        get_embeddings_from_set(model, WORDS, preprocessors=None)

    with pytest.raises(
        TypeError,
        match=(
            r"preprocessors must indicate at least one preprocessor, even if it is "
            r"an empty dictionary {}, got: .*\."
        ),
    ):
        get_embeddings_from_set(model, WORDS, preprocessors=[])

    with pytest.raises(
        TypeError, match=r"each preprocessor should be a dict, got .* at index .*\."
    ):
        get_embeddings_from_set(
            model, WORDS, preprocessors=[{"lower": True}, {"upper": True}, 1]
        )

    with pytest.raises(
        ValueError, match=r"strategy should be 'first' or 'all', got .*\."
    ):
        get_embeddings_from_set(model, WORDS, strategy=None)

    with pytest.raises(
        ValueError, match=r"strategy should be 'first' or 'all', got .*\."
    ):
        get_embeddings_from_set(model, WORDS, strategy="blabla")


def test_get_embeddings_from_word_set(model):

    # ----------------------------------------------------------------------------------
    # test basic operation of _get_embeddings_from_word_set
    WORDS = ["man", "woman"]

    not_found_words, embeddings = get_embeddings_from_set(model, WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test with a word that does not exists in the model
    WORDS = ["man", "woman", "not_a_word_"]
    not_found_words, embeddings = get_embeddings_from_set(model, WORDS)

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
    not_found_words, embeddings = get_embeddings_from_set(
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
    not_found_words, embeddings = get_embeddings_from_set(
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
    not_found_words, embeddings = get_embeddings_from_set(
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
    assert not_found_words == ["WoMan"]

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test two word preprocessors strip_accents strategy="all":
    WORDS = [
        "mán",
        "WöMàn",
    ]
    not_found_words, embeddings = get_embeddings_from_set(
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
    assert not_found_words == ["WoMan"]

    assert [np.array_equal(model[k], embeddings[k]) for k in embeddings.keys()]

    # ----------------------------------------------------------------------------------
    # test normalize
    WORDS = ["man", "woman"]

    _, embeddings = get_embeddings_from_set(model, WORDS, normalize=True)

    assert 0.99999 < np.linalg.norm(embeddings["man"]) < 1.00001
    assert 0.99999 < np.linalg.norm(embeddings["woman"]) < 1.00001


# -------------------------------------------------------------------------------------
def test_get_embeddings_from_sets(model, caplog, capsys):

    # ----------------------------------------------------------------------------------
    # Test types and value checking.

    with pytest.raises(
        TypeError,
        match=(
            r"sets should be a sequence of sequences \(list, tuple or np\.array\) of "
            r"strings, got:.*"
        ),
    ):
        get_embeddings_from_sets(model, None)

    with pytest.raises(
        TypeError,
        match=(
            r"Every set in sets should be a list, tuple or np.array of strings"
            r", got in index.*"
        ),
    ):
        get_embeddings_from_sets(model, [None])

    with pytest.raises(
        TypeError,
        match=(
            r"All set elements in a set of words should be strings. "
            r"Got in set.*at position 0:.*"
        ),
    ):
        get_embeddings_from_sets(model, [[1, "he"]])

    with pytest.raises(
        TypeError,
        match=(
            r"All set elements in a set of words should be strings. "
            r"Got in set.* at position 1:.*"
        ),
    ):
        get_embeddings_from_sets(model, [["she", 1]])

    with pytest.raises(
        TypeError, match=r"sets_name should be a string or None, got:.*",
    ):
        get_embeddings_from_sets(model, [["she", "he"]], 0)

    with pytest.raises(
        TypeError, match=r"warn_lost_sets should be a bool, got:.*",
    ):
        get_embeddings_from_sets(
            model, [["she", "he"]], "definning", warn_lost_sets=None
        )

    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got:.*",
    ):
        get_embeddings_from_sets(
            model, [["she", "he"]], "definning", True, verbose=None
        )

    # ----------------------------------------------------------------------------------
    # Test with pairs of words

    pairs = [["woman", "man"], ["she", "he"], ["mother", "father"]]
    pairs_set_name = "definning"

    embedding_pairs = get_embeddings_from_sets(
        model, sets=pairs, sets_name=pairs_set_name
    )

    assert isinstance(embedding_pairs, list)

    for embedding_pair in embedding_pairs:
        assert isinstance(embedding_pair, dict)
        assert len(embedding_pair.keys()) == 2
        assert len(embedding_pair.values()) == 2
        for w, e in embedding_pair.items():
            assert isinstance(w, str)
            assert isinstance(e, np.ndarray)
            assert e.shape == (300,)
            assert all(model[w] == e)

    # ----------------------------------------------------------------------------------
    # Test with sets

    sets = [
        ["judaism", "christianity", "islam"],
        ["jew", "christian", "muslim"],
        ["synagogue", "church", "mosque"],
    ]
    sets_name = "definning"

    embedding_pairs = get_embeddings_from_sets(model, sets=sets, sets_name=sets_name)

    assert isinstance(embedding_pairs, list)

    for embedding_pair in embedding_pairs:
        assert isinstance(embedding_pair, dict)
        assert len(embedding_pair.keys()) == 3
        assert len(embedding_pair.values()) == 3
        for w, e in embedding_pair.items():
            assert isinstance(w, str)
            assert isinstance(e, np.ndarray)
            assert e.shape == (300,)
            assert all(model[w] == e)

    # ----------------------------------------------------------------------------------
    # Test out of vocabulary (OOV) words and failures

    oov_pairs = [["the", "vbbge"], ["ddsds", "ferhh"]]
    pairs_with_oov = pairs + oov_pairs

    with caplog.at_level(logging.INFO):
        embedding_pairs_2 = get_embeddings_from_sets(
            model,
            sets=pairs_with_oov,
            sets_name=pairs_set_name,
            warn_lost_sets=True,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert len(embedding_pairs_2) == 3
        assert "Word(s) found: ['the'], not found: ['vbbge']" in out
        assert "Word(s) found: [], not found: ['ddsds', 'ferhh']" in out

        assert "3/5 sets of words were correctly converted to sets of embeddings" in out

    with pytest.raises(
        Exception,
        match=r"No set could be converted to embedding because no set "
        "could be fully found in the model vocabulary.",
    ):
        get_embeddings_from_sets(model, sets=oov_pairs)


#
def test_warn_not_found_words(caplog):

    with pytest.raises(
        TypeError, match=r"warn_not_found_words should be a boolean, got .*\."
    ):
        _warn_not_found_words(None, [], "", "")

    _warn_not_found_words(True, ["aaa", "bbb"], "some_model", "set1")
    msg = (
        "The following words from set 'set1' do not exist within the vocabulary "
        "of some_model: ['aaa', 'bbb']"
    )
    assert msg in caplog.text


def test_get_embeddings_from_query(caplog, simple_query, model: WordEmbeddingModel):
    query, flowers, insects, pleasant, unpleasant = simple_query
    w2v = model.wv
    # test types

    # target sets None
    with pytest.raises(TypeError, match="query should be an instance of Query, got"):
        get_embeddings_from_query(model, None)

    with pytest.raises(
        TypeError,
        match=r"preprocessors should be a list of dicts which contains preprocessor options, got .*\.",
    ):
        get_embeddings_from_query(model, query, preprocessors=1)

    with pytest.raises(
        TypeError, match="warn_not_found_words should be a boolean, got"
    ):
        get_embeddings_from_query(model, query, warn_not_found_words=None)

    embeddings = get_embeddings_from_query(model, query)

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
    assert target_embeddings_names[0] == "Flowers"
    assert target_embeddings_names[1] == "Insects"
    assert attribute_embeddings_names[0] == "Pleasant"
    assert attribute_embeddings_names[1] == "Unpleasant"

    # test set embeddings
    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != "axe", insects)
    )
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]["aster"] == w2v["aster"])
    assert list(target_embeddings_sets[1]["ant"] == w2v["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == w2v["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == w2v["abuse"])


def test_preprocessor_args_on_get_embeddings_from_query(caplog, simple_query, model):
    _, flowers, insects, pleasant, unpleasant = simple_query
    w2v = model.wv

    # with lost words and warn_not_found_words=True
    flowers_2 = flowers + ["aaa", "bbb"]
    query_2 = Query(
        [flowers_2, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query_2, warn_not_found_words=True)
    assert (
        "The following words from set 'Flowers' do not exist within the vocabulary"
        in caplog.text
    )
    assert "['aaa', 'bbb']" in caplog.text

    # with preprocessor options
    flowers_3 = [s.upper() for s in flowers]
    query_3 = Query(
        [flowers_3, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(
        model, query_3, preprocessors=[{"lowercase": True}]
    )

    target_embeddings, attribute_embeddings = embeddings

    target_embeddings_sets = list(target_embeddings.values())
    attribute_embeddings_sets = list(attribute_embeddings.values())

    assert len(target_embeddings_sets) == 2
    assert len(attribute_embeddings_sets) == 2

    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != "axe", insects)
    )
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]["aster"] == w2v["aster"])
    assert list(target_embeddings_sets[1]["ant"] == w2v["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == w2v["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == w2v["abuse"])

    # with secondary_preprocessor_options options
    embeddings = get_embeddings_from_query(
        model, query_3, preprocessors=[{}, {"lowercase": True}]
    )

    target_embeddings, attribute_embeddings = embeddings

    target_embeddings_sets = list(target_embeddings.values())
    attribute_embeddings_sets = list(attribute_embeddings.values())

    assert len(target_embeddings_sets) == 2
    assert len(attribute_embeddings_sets) == 2

    assert list(target_embeddings_sets[0].keys()) == flowers
    assert list(target_embeddings_sets[1].keys()) == list(
        filter(lambda x: x != "axe", insects)
    )
    assert list(attribute_embeddings_sets[0].keys()) == pleasant
    assert list(attribute_embeddings_sets[1].keys()) == unpleasant

    assert list(target_embeddings_sets[0]["aster"] == w2v["aster"])
    assert list(target_embeddings_sets[1]["ant"] == w2v["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == w2v["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == w2v["abuse"])


def test_threshold_param_on_get_embeddings_from_query(caplog, simple_query, model):
    query, flowers, insects, pleasant, unpleasant = simple_query

    # check param type
    with pytest.raises(
        TypeError, match=r"lost_vocabulary_threshold should be float, .*"
    ):
        get_embeddings_from_query(model, query, lost_vocabulary_threshold="")

    # with lost vocabulary threshold.
    flowers_ = flowers + ["aaa", "aab", "aac", "aad", "aaf", "aag", "aah", "aai", "aaj"]
    query = Query(
        [flowers_, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Flowers' into" in caplog.text

    # with lost vocabulary threshold.
    insects_ = insects + ["aaa", "aab", "aac", "aad", "aaf", "aag", "aah", "aai", "aaj"]
    query = Query(
        [flowers, insects_],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Insects' into" in caplog.text

    # with lost vocabulary threshold.
    pleasant_ = pleasant + [
        "aaa",
        "aab",
        "aac",
        "aad",
        "aaf",
        "aag",
        "aah",
        "aai",
        "aaj",
    ]
    query = Query(
        [flowers, insects],
        [pleasant_, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Pleasant' into" in caplog.text

    # test attribute 2 with lost vocabulary threshold.
    unpleasant_ = insects + [
        "aaa",
        "aab",
        "aac",
        "aad",
        "aaf",
        "aag",
        "aah",
        "aai",
        "aaj",
    ]
    query = Query(
        [flowers, insects],
        [pleasant, unpleasant_],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Unpleasant' into" in caplog.text

    # with lost vocabulary threshold.
    unpleasant_ = insects + [
        "aaa",
        "aab",
        "aac",
        "aad",
        "aaf",
        "aag",
        "aah",
        "aai",
        "aaj",
    ]
    query = Query(
        [flowers, insects],
        [pleasant, unpleasant_],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(model, query, lost_vocabulary_threshold=0.5)

    assert embeddings is not None

