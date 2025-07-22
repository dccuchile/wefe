import logging

import numpy as np
import pytest

from wefe.datasets.datasets import load_weat
from wefe.preprocessing import (
    _warn_not_found_words,
    get_embeddings_from_query,
    get_embeddings_from_set,
    get_embeddings_from_tuples,
    preprocess_word,
)
from wefe.query import Query
from wefe.utils import load_test_model
from wefe.word_embedding_model import WordEmbeddingModel


@pytest.fixture
def model() -> WordEmbeddingModel:
    """Load a subset of Word2vec as a testing model.

    Returns
    -------
    WordEmbeddingModel
        The loaded testing model.

    """
    return load_test_model()


@pytest.fixture
def query_2t2a_1(weat_wordsets: dict[str, list[str]]) -> Query:
    """Generate a Flower and Insects wrt Pleasant vs Unpleasant test query.

    Parameters
    ----------
    weat_wordsets : Dict[str, List[str]]
        The word sets used in WEAT original work.

    Returns
    -------
    Query
        The generated query.

    """
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    return query


@pytest.fixture
def weat_wordsets() -> dict[str, list[str]]:
    """Load the word sets used in WEAT original work.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary that map a word set name to a set of words.

    """
    weat_wordsets = load_weat()
    return weat_wordsets


@pytest.fixture
def query_2t2a_uppercase(weat_wordsets: dict[str, list[str]]) -> Query:
    """Generate a Flower and Insects wrt Pleasant vs Unpleasant test query.

    Parameters
    ----------
    weat_wordsets : Dict[str, List[str]]
        The word sets used in WEAT original work.

    Returns
    -------
    Query
        The generated query.

    """
    query = Query(
        [
            [s.upper() for s in weat_wordsets["flowers"]],
            [s.upper() for s in weat_wordsets["insects"]],
        ],
        [
            [s.upper() for s in weat_wordsets["pleasant_5"]],
            [s.upper() for s in weat_wordsets["unpleasant_5"]],
        ],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    return query


# --------------------------------------------------------------------------------------
# test preprocess_word
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("input_word", "preprocessor_opts", "vocab_prefix", "expected"),
    argvalues=[
        ("Woman", {}, None, "Woman"),
        ("Woman", {"lowercase": True}, None, "woman"),
        ("Woman", {"uppercase": True}, None, "WOMAN"),
        ("Woman", {"titlecase": True}, None, "Woman"),
        ("Woman", {"lowercase": True, "case": True, "title": True}, None, "woman"),
        ("wömàn", {"strip_accents": True}, None, "woman"),
        ("wömàn", {"strip_accents": "ascii"}, None, "woman"),
        ("wömàn", {"strip_accents": "unicode"}, None, "woman"),
        ("WöMàn", {"lowercase": True, "strip_accents": True}, None, "woman"),
        ("WöMàn", {"uppercase": True, "strip_accents": True}, None, "WOMAN"),
        ("WöMàn", {"titlecase": True, "strip_accents": True}, None, "Woman"),
        ("Woman", {"preprocessor": lambda x: x.lower()}, None, "woman"),
        (
            "Woman",
            {"preprocessor": lambda x: x.upper(), "lowercase": True},
            None,
            "WOMAN",
        ),
        ("woman", {}, "asd-", "asd-woman"),
    ],
    ids=[
        "no_preprocessor",
        "lowercase",
        "uppercase",
        "titlecase",
        "multiple_case_options",
        "strip_accents_true",
        "strip_accents_ascii",
        "strip_accents_unicode",
        "lowercase_and_strip_accents",
        "uppercase_and_strip_accents",
        "titlecase_and_strip_accents",
        "custom_preprocessor_lower",
        "custom_preprocessor_upper_overrides_lower",
        "with_vocab_prefix",
    ],
)
def test_preprocess_word(input_word, preprocessor_opts, vocab_prefix, expected) -> None:
    result = preprocess_word(input_word, preprocessor_opts, vocab_prefix=vocab_prefix)
    assert result == expected


# --------------------------------------------------------------------------------------
# test get_embeddings_from_set
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=(
        "model_",
        "word_set",
        "preprocessors",
        "strategy",
        "expected_exception",
        "match",
    ),
    argvalues=[
        (
            None,
            None,
            [{}],
            None,
            TypeError,
            r"model should be a WordEmbeddingModel instance, got .*\.",
        ),
        (
            "model",  # will be replaced in test body
            None,
            [{}],
            None,
            TypeError,
            r"word_set should be a list, tuple or np.array of strings, got.*",
        ),
        (
            "model",
            ["man", "woman"],
            None,
            None,
            TypeError,
            (
                r"preprocessors should be a list of dicts which contains preprocessor "
                r"options, got .*\."
            ),
        ),
        (
            "model",
            ["man", "woman"],
            [],
            None,
            TypeError,
            (
                r"preprocessors must indicate at least one preprocessor, even if it is "
                r"an empty dictionary {}, got: .*\."
            ),
        ),
        (
            "model",
            ["man", "woman"],
            [{"lower": True}, {"upper": True}, 1],
            None,
            TypeError,
            r"each preprocessor should be a dict, got .* at index .*\.",
        ),
        (
            "model",
            ["man", "woman"],
            [{}],
            "blabla",
            ValueError,
            r"strategy should be 'first' or 'all', got .*\.",
        ),
    ],
    ids=[
        "model_none",
        "word_set_none",
        "preprocessors_none",
        "preprocessors_empty",
        "preprocessor_not_dict",
        "strategy_invalid",
    ],
)
def test_get_embeddings_from_set_types(
    model_,
    word_set,
    preprocessors,
    strategy,
    expected_exception,
    match,
    request,
) -> None:
    kwargs = {
        "model": request.getfixturevalue("model") if model_ == "model" else model_,
        "word_set": word_set,
        "preprocessors": preprocessors,
    }
    if strategy is not None:
        kwargs["strategy"] = strategy

    with pytest.raises(expected_exception, match=match):
        get_embeddings_from_set(**kwargs)


def test_get_embeddings_from_set(model: WordEmbeddingModel) -> None:
    # ----------------------------------------------------------------------------------
    # test basic operation of get_embeddings_from_set
    WORDS = ["man", "woman"]

    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[{}],
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])


def test_get_embeddings_from_set_with_oov(model: WordEmbeddingModel) -> None:
    # test with a word that does not exists in the model
    WORDS = ["man", "woman", "not_a_word_"]
    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[{}],
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 1

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == ["not_a_word_"]

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])


def test_get_embeddings_from_set_prep_lowercase(model: WordEmbeddingModel) -> None:
    # test word preprocessor lowercase

    WORDS = [
        "mAN",
        "WOmaN",
    ]
    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[{"lowercase": True}],
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])


def test_get_embeddings_from_set_prep_strip_accents(model: WordEmbeddingModel) -> None:
    # test word preprocessor strip_accents:
    WORDS = [
        "mán",
        "wömàn",
    ]
    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[{"strip_accents": True}],
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(model["man"], embeddings["man"])
    assert np.array_equal(model["woman"], embeddings["woman"])


def test_get_embeddings_from_set_prep_strategy_first(model: WordEmbeddingModel) -> None:
    # test two word preprocessors strip_accents strategy="first":
    WORDS = [
        "mán",
        "WöMàn",
    ]
    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[
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


def test_get_embeddings_from_set_prep_strategy_all(model: WordEmbeddingModel) -> None:
    # test two word preprocessors strip_accents strategy="all":
    WORDS = [
        "mán",
        "WöMàn",
    ]
    not_found_words, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[
            {"strip_accents": True},
            {"strip_accents": True, "lowercase": True},
            {"strip_accents": True, "uppercase": True},
            {"strip_accents": True, "titlecase": True},
        ],
        strategy="all",
    )

    assert list(embeddings.keys()) == ["man", "MAN", "Man", "woman", "WOMAN", "Woman"]
    assert not_found_words == ["WoMan"]

    assert [np.array_equal(model[k], embeddings[k]) for k in embeddings]


def test_get_embeddings_from_set_with_normalization(model: WordEmbeddingModel) -> None:
    # test normalize
    WORDS = ["man", "woman"]

    _, embeddings = get_embeddings_from_set(
        model=model,
        word_set=WORDS,
        preprocessors=[{}],
        normalize=True,
    )

    assert 0.99999 < np.linalg.norm(embeddings["man"]) < 1.00001
    assert 0.99999 < np.linalg.norm(embeddings["woman"]) < 1.00001


# --------------------------------------------------------------------------------------
# test get_embeddings_from_sets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=(
        "model_",
        "sets",
        "sets_name",
        "preprocessors",
        "warn_lost_sets",
        "verbose",
        "match",
    ),
    argvalues=[
        (
            None,
            [["he"]],
            None,
            [{}],
            False,
            False,
            r"model should be a WordEmbeddingModel instance, got <class 'NoneType'>.",
        ),
        (
            "model",  # will be replaced in test body
            None,
            None,
            [{}],
            False,
            False,
            (
                r"sets should be a sequence of sequences "
                r"\(list, tuple or np\.array\) of strings, got:.*"
            ),
        ),
        (
            "model",
            [None],
            None,
            [{}],
            False,
            False,
            (
                r"Every set in sets should be a list, tuple or np.array of strings, "
                "got in index.*"
            ),
        ),
        (
            "model",
            [[1, "he"]],
            None,
            [{}],
            False,
            False,
            (
                r"All set elements in a set of words should be strings. "
                "Got in set.*at position 0:.*"
            ),
        ),
        (
            "model",
            [["she", 1]],
            None,
            [{}],
            False,
            False,
            (
                r"All set elements in a set of words should be strings. "
                "Got in set.* at position 1:.*"
            ),
        ),
        (
            "model",
            [["she", "he"]],
            0,
            [{}],
            False,
            False,
            r"sets_name should be a string or None, got:.*",
        ),
        (
            "model",
            [["she", "he"]],
            "definning",
            [{}],
            None,
            False,
            r"warn_lost_sets should be a bool, got:.*",
        ),
        (
            "model",
            [["she", "he"]],
            "definning",
            [{}],
            True,
            "yes",
            r"verbose should be a bool, got:.*",
        ),
        (
            "model",
            [["she", "he"]],
            "definning",
            [{}],
            True,
            None,
            r"verbose should be a bool, got:.*",
        ),
    ],
    ids=[
        "model_none",
        "sets_none",
        "set_element_none",
        "set_element_not_str_0",
        "set_element_not_str_1",
        "sets_name_not_str",
        "warn_lost_sets_not_bool",
        "verbose_not_bool_str",
        "verbose_not_bool_none",
    ],
)
def test_get_embeddings_from_sets_type_checkings(
    model_,
    sets,
    sets_name,
    preprocessors,
    warn_lost_sets,
    verbose,
    match,
    request,
):
    kwargs = {
        "model": request.getfixturevalue("model") if model_ == "model" else model_,
        "sets": sets,
        "preprocessors": preprocessors,
    }
    if sets_name is not None:
        kwargs["sets_name"] = sets_name
    if warn_lost_sets is not False:
        kwargs["warn_lost_sets"] = warn_lost_sets
    if verbose is not False:
        kwargs["verbose"] = verbose

    with pytest.raises(TypeError, match=match):
        get_embeddings_from_tuples(**kwargs)


def test_get_embeddings_from_sets_with_monuples(model: WordEmbeddingModel) -> None:
    # Test with 1-tuples

    pairs = [["woman"], ["she"], ["mother"]]
    pairs_set_name = "definning"

    embedding_pairs = get_embeddings_from_tuples(
        model,
        sets=pairs,
        sets_name=pairs_set_name,
        preprocessors=[{}],
    )

    assert isinstance(embedding_pairs, list)

    for embedding_pair in embedding_pairs:
        assert isinstance(embedding_pair, dict)
        assert len(embedding_pair.keys()) == 1
        assert len(embedding_pair.values()) == 1

        for word, embedding in embedding_pair.items():
            assert isinstance(word, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (300,)
            assert all(model[word] == embedding)


def test_get_embeddings_from_sets_with_pairs(model: WordEmbeddingModel) -> None:
    # Test with pairs of words (2-tuples)

    pairs = [["woman", "man"], ["she", "he"], ["mother", "father"]]
    pairs_set_name = "definning"

    embedding_pairs = get_embeddings_from_tuples(
        model,
        sets=pairs,
        sets_name=pairs_set_name,
        preprocessors=[{}],
    )

    assert isinstance(embedding_pairs, list)

    for embedding_pair in embedding_pairs:
        assert isinstance(embedding_pair, dict)
        assert len(embedding_pair.keys()) == 2
        assert len(embedding_pair.values()) == 2

        for word, embedding in embedding_pair.items():
            assert isinstance(word, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (300,)
            assert all(model[word] == embedding)


def test_get_embeddings_from_sets_with_triple(model: WordEmbeddingModel) -> None:
    # Test with 3-tuples

    sets = [
        ["judaism", "christianity", "islam"],
        ["jew", "christian", "muslim"],
        ["synagogue", "church", "mosque"],
    ]
    sets_name = "definning"

    embedding_pairs = get_embeddings_from_tuples(
        model=model,
        sets=sets,
        sets_name=sets_name,
        preprocessors=[{}],
    )

    assert isinstance(embedding_pairs, list)

    for embedding_pair in embedding_pairs:
        assert isinstance(embedding_pair, dict)
        assert len(embedding_pair.keys()) == 3
        assert len(embedding_pair.values()) == 3

        for word, embedding in embedding_pair.items():
            assert isinstance(word, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (300,)
            assert all(model[word] == embedding)


def test_get_embeddings_from_sets_with_oov(
    model: WordEmbeddingModel,
    caplog,
    capsys,
) -> None:
    # Test out of vocabulary (OOV) words and failures

    pairs = [["woman", "man"], ["she", "he"], ["mother", "father"]]
    pairs_set_name = "definning"

    oov_pairs = [["the", "vbbge"], ["ddsds", "ferhh"]]
    pairs_with_oov = pairs + oov_pairs

    with caplog.at_level(logging.INFO):
        embedding_pairs_2 = get_embeddings_from_tuples(
            model,
            sets=pairs_with_oov,
            sets_name=pairs_set_name,
            warn_lost_sets=True,
            preprocessors=[{}],
            verbose=True,
        )
        out = capsys.readouterr().out
        assert len(embedding_pairs_2) == 3
        assert "Word(s) found: ['the'], not found: ['vbbge']" in out
        assert "Word(s) found: [], not found: ['ddsds', 'ferhh']" in out

        assert "3/5 sets of words were correctly converted to sets of embeddings" in out


def test_get_embeddings_from_sets_with_no_set_converted(
    model: WordEmbeddingModel,
) -> None:
    oov_pairs = [["the", "vbbge"], ["ddsds", "ferhh"]]

    with pytest.raises(
        Exception,
        match=r"No set could be converted to embedding because no set "
        "could be fully found in the model vocabulary.",
    ):
        get_embeddings_from_tuples(
            model=model,
            sets=oov_pairs,
            preprocessors=[{}],
        )


# --------------------------------------------------------------------------------------
# test warn_not_found_words
# --------------------------------------------------------------------------------------


def test_warn_not_found_words(caplog) -> None:
    with pytest.raises(
        TypeError, match=r"warn_not_found_words should be a boolean, got .*\."
    ):
        _warn_not_found_words(
            warn_not_found_words=None,
            not_found_words=[],
            model_name="",
            set_name="",
        )

    _warn_not_found_words(True, ["aaa", "bbb"], "some_model", "set1")
    msg = (
        "The following words from set 'set1' do not exist within the vocabulary "
        "of some_model: ['aaa', 'bbb']"
    )
    assert msg in caplog.text


# --------------------------------------------------------------------------------------
# test get_embeddings_from_sets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "query",
        "preprocessors",
        "lost_vocabulary_threshold",
        "warn_not_found_words",
        "expected_exception",
        "match",
    ),
    [
        (
            None,
            [{}],
            0.2,
            False,
            TypeError,
            "query should be an instance of Query, got",
        ),
        (
            "query",
            1,
            0.2,
            False,
            TypeError,
            (
                r"preprocessors should be a list of dicts which contains preprocessor "
                r"options, got .*\."
            ),
        ),
        (
            "query",
            [{}],
            0.2,
            None,
            TypeError,
            "warn_not_found_words should be a boolean, got",
        ),
        (
            "query",
            [{}],
            "",
            False,
            TypeError,
            r"lost_vocabulary_threshold should be float, .*",
        ),
    ],
    ids=[
        "query_none",
        "preprocessors_not_list",
        "warn_not_found_words_not_bool",
        "lost_vocabulary_threshold_not_float",
    ],
)
def test_get_embeddings_from_query_input_checking_parametrized(
    query: str | None,
    preprocessors,
    lost_vocabulary_threshold,
    warn_not_found_words,
    expected_exception,
    match,
    request: pytest.FixtureRequest,
) -> None:
    query = request.getfixturevalue("query_2t2a_1") if query == "query" else query
    model = request.getfixturevalue("model")

    kwargs = {
        "model": model,
        "query": query,
        "preprocessors": preprocessors,
        "lost_vocabulary_threshold": lost_vocabulary_threshold,
    }
    if warn_not_found_words is not False:
        kwargs["warn_not_found_words"] = warn_not_found_words

    with pytest.raises(expected_exception, match=match):
        get_embeddings_from_query(**kwargs)


def test_get_embeddings_from_query(
    query_2t2a_1: Query, weat_wordsets: dict[str, list[str]], model: WordEmbeddingModel
) -> None:
    flowers, insects, pleasant, unpleasant = (
        weat_wordsets["flowers"],
        weat_wordsets["insects"],
        weat_wordsets["pleasant_5"],
        weat_wordsets["unpleasant_5"],
    )

    word_vectors = model.wv

    embeddings = get_embeddings_from_query(
        model=model,
        query=query_2t2a_1,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.2,
    )

    assert embeddings is not None
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

    assert list(target_embeddings_sets[0]["aster"] == word_vectors["aster"])
    assert list(target_embeddings_sets[1]["ant"] == word_vectors["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == word_vectors["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == word_vectors["abuse"])


def test_get_embeddings_from_query_oov_warns(
    caplog,
    model: WordEmbeddingModel,
    weat_wordsets: dict[str, list[str]],
) -> None:
    # check lost words warning when warn_not_found_words is True

    flowers, insects, pleasant, unpleasant = (
        weat_wordsets["flowers"],
        weat_wordsets["insects"],
        weat_wordsets["pleasant_5"],
        weat_wordsets["unpleasant_5"],
    )

    flowers_with_oov = flowers + ["aaa", "bbb"]
    query_with_oov_1 = Query(
        [flowers_with_oov, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )

    embeddings = get_embeddings_from_query(
        model=model,
        query=query_with_oov_1,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.2,
        warn_not_found_words=True,
    )
    assert embeddings is not None
    assert (
        "The following words from set 'Flowers' do not exist within the vocabulary"
        in caplog.text
    )
    assert "['aaa', 'bbb']" in caplog.text


def test_get_embeddings_from_query_with_lower_preprocessor(
    model: WordEmbeddingModel,
    query_2t2a_uppercase: Query,
    weat_wordsets: dict[str, list[str]],
) -> None:
    # check get_embeddings_from_query with lowercase and one preprocessor options
    flowers, insects, pleasant, unpleasant = (
        weat_wordsets["flowers"],
        weat_wordsets["insects"],
        weat_wordsets["pleasant_5"],
        weat_wordsets["unpleasant_5"],
    )

    embeddings = get_embeddings_from_query(
        model=model,
        query=query_2t2a_uppercase,
        preprocessors=[{"lowercase": True}],
        lost_vocabulary_threshold=0.2,
    )

    assert embeddings is not None
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

    assert list(target_embeddings_sets[0]["aster"] == model["aster"])
    assert list(target_embeddings_sets[1]["ant"] == model["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == model["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == model["abuse"])


def test_get_embeddings_from_query_with_two_preprocessors(
    model: WordEmbeddingModel,
    query_2t2a_uppercase: Query,
    weat_wordsets: dict[str, list[str]],
) -> None:
    # test get_embeddings_from_query with secondary preprocessor_options options
    flowers, insects, pleasant, unpleasant = (
        weat_wordsets["flowers"],
        weat_wordsets["insects"],
        weat_wordsets["pleasant_5"],
        weat_wordsets["unpleasant_5"],
    )
    embeddings = get_embeddings_from_query(
        model=model,
        query=query_2t2a_uppercase,
        preprocessors=[{}, {"lowercase": True}],
        lost_vocabulary_threshold=0.2,
    )

    assert embeddings is not None
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

    assert list(target_embeddings_sets[0]["aster"] == model["aster"])
    assert list(target_embeddings_sets[1]["ant"] == model["ant"])
    assert list(attribute_embeddings_sets[0]["caress"] == model["caress"])
    assert list(attribute_embeddings_sets[1]["abuse"] == model["abuse"])


def test_get_embeddings_from_query_lost_threshold(
    caplog, model: WordEmbeddingModel, weat_wordsets: dict[str, list[str]]
) -> None:
    flowers, insects, pleasant, unpleasant = (
        weat_wordsets["flowers"],
        weat_wordsets["insects"],
        weat_wordsets["pleasant_5"],
        weat_wordsets["unpleasant_5"],
    )

    # with lost vocabulary threshold.
    flowers_ = flowers + ["aaa", "aab", "aac", "aad", "aaf", "aag", "aah", "aai", "aaj"]
    query = Query(
        [flowers_, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.1,
    )

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
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.1,
    )

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
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.1,
    )

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
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.1,
    )

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
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        preprocessors=[{}],
        lost_vocabulary_threshold=0.5,
    )

    assert embeddings is not None
