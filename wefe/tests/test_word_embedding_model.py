import gensim
import pytest
import numpy as np
import logging

from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText, KeyedVectors
import semantic_version

from ..query import Query
from ..datasets import load_weat
from ..word_embedding_model import WordEmbeddingModel
from ..utils import load_weat_w2v

LOGGER = logging.getLogger(__name__)
gensim_version = semantic_version.Version.coerce(gensim.__version__)


@pytest.fixture
def word2vec_sm():
    w2v = KeyedVectors.load_word2vec_format("./wefe/tests/w2v_sm.bin", binary=True)
    return WordEmbeddingModel(w2v, "word2vec")


def test__init__(word2vec_sm):

    # Test types verifications

    # target sets None
    with pytest.raises(
        TypeError, match="model should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel(None)

    # target sets int
    with pytest.raises(
        TypeError, match="model should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel("abc")

    with pytest.raises(
        TypeError, match="model should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel(1)

    with pytest.raises(
        TypeError, match="model should be an instance of gensim's BaseKeyedVectors"
    ):
        WordEmbeddingModel({})

    w2v = KeyedVectors.load_word2vec_format("./wefe/tests/w2v_sm.bin", binary=True)

    # test models
    model = WordEmbeddingModel(w2v)
    assert model.model == w2v
    assert model.model_name == "Unnamed word embedding model"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(w2v, "w2v_sm")
    assert model.model == w2v
    assert model.model_name == "w2v_sm"
    assert model.vocab_prefix is None

    model = WordEmbeddingModel(w2v, "w2v_sm", "\\c\\en")
    assert model.model == w2v
    assert model.model_name == "w2v_sm"
    assert model.vocab_prefix == "\\c\\en"


def test__eq__(word2vec_sm):
    model_1 = WordEmbeddingModel(word2vec_sm.model, "w2v")
    model_2 = WordEmbeddingModel(word2vec_sm.model, "w2v_2")
    model_3 = WordEmbeddingModel(word2vec_sm.model, "w2v_3", vocab_prefix="a")

    assert model_1 == model_1
    assert model_1 != model_2

    model_1.model = None

    assert model_1 != model_2

    assert model_2 != model_3
    assert model_3 == model_3


def test__getitem__(word2vec_sm):

    embedding = word2vec_sm["ASDF"]
    assert embedding is None

    embedding = word2vec_sm["career"]
    assert isinstance(embedding, np.ndarray)


def test__init__with_w2v_model():

    if gensim_version.major >= 4:
        w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=-1)
    else:
        w2v = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=-1)
    w2v_keyed_vectors = w2v.wv
    wem = WordEmbeddingModel(w2v_keyed_vectors, "w2v")

    assert w2v.wv == wem.model


def test__init_with_fast_model():

    if gensim_version.major >= 4:
        fast = FastText(
            vector_size=4, window=3, min_count=1, sentences=common_texts, epochs=10
        )
    else:
        fast = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)
    fast_keyed_vectors = fast.wv
    wem = WordEmbeddingModel(fast_keyed_vectors, "w2v")

    assert fast.wv == wem.model


def test__preprocess_word(word2vec_sm):

    word = word2vec_sm._preprocess_word("Woman")
    assert word == "Woman"

    word = word2vec_sm._preprocess_word("Woman", {"lowercase": True})
    assert word == "woman"

    word = word2vec_sm._preprocess_word("wömàn", {"strip_accents": True})
    assert word == "woman"

    word = word2vec_sm._preprocess_word("wömàn", {"strip_accents": "ascii"})
    assert word == "woman"

    word = word2vec_sm._preprocess_word("wömàn", {"strip_accents": "unicode"})
    assert word == "woman"

    # all together
    word = word2vec_sm._preprocess_word(
        "WöMàn", {"lowercase": True, "strip_accents": True}
    )
    assert word == "woman"

    # check for custom preprocessor
    word = word2vec_sm._preprocess_word("Woman", {"preprocessor": lambda x: x.lower()})
    assert word == "woman"

    # check if preprocessor overrides any other option
    word = word2vec_sm._preprocess_word(
        "Woman", {"preprocessor": lambda x: x.upper(), "lowercase": True}
    )
    assert word == "WOMAN"

    # now with prefix
    word2vec_sm = WordEmbeddingModel(word2vec_sm.model, "weat_w2v", "asd-")
    word = word2vec_sm._preprocess_word("woman")
    assert word == "asd-woman"


# -------------------------------------------------------------------------------------
def test_normalize_embeddings(word2vec_sm):
    word2vec_sm.normalize_embeddings()
    for word in word2vec_sm.vocab:
        assert np.linalg.norm(word2vec_sm[word]) - 1.0 < 0.000001

    word2vec_sm.model = None

    with pytest.raises(
        TypeError, match="The model does not have the init_sims method implemented."
    ):
        word2vec_sm.normalize_embeddings()


# -------------------------------------------------------------------------------------
def test_update_embedding(word2vec_sm):
    new_embedding = np.ones(300, dtype=word2vec_sm.model.vectors.dtype)
    word2vec_sm.update_embedding("The", new_embedding)
    assert all(word2vec_sm["The"] == new_embedding)

    with pytest.raises(TypeError, match=r"word should be a string, got .*"):
        word2vec_sm.update_embedding(0, new_embedding)

    with pytest.raises(ValueError, match=r"word .* not in model vocab."):
        word2vec_sm.update_embedding("blablablablabla", new_embedding)

    with pytest.raises(
        TypeError, match=r".* new embedding should be a np\.array, got .*\."
    ):
        word2vec_sm.update_embedding("The", 0)

    with pytest.raises(
        ValueError,
        match=r"The size of .* embedding (.*) is different from the size of the "
        r"embeddings in the model (.*)\.",
    ):
        word2vec_sm.update_embedding("The", np.ones(200, dtype=np.float64))

    with pytest.raises(
        ValueError, match=r"embedding dtype .* is not the same of model's dtype .*\."
    ):
        word2vec_sm.update_embedding("The", np.ones(300, dtype=np.float64))


# -------------------------------------------------------------------------------------
def test_update_embeddings(word2vec_sm):
    words = ["The", "in"]
    embeddings = [np.ones(300, dtype=np.float32), np.ones(300, dtype=np.float32) * -1]

    word2vec_sm.update_embeddings(words, embeddings)

    assert all(word2vec_sm["The"] == embeddings[0])
    assert all(word2vec_sm["in"] == embeddings[1])

    embeddings_in_array = np.array(
        [np.ones(300, dtype=np.float32) * 2, np.ones(300, dtype=np.float32) * -2]
    )
    word2vec_sm.update_embeddings(words, embeddings_in_array)

    assert all(word2vec_sm["The"] == embeddings_in_array[0])
    assert all(word2vec_sm["in"] == embeddings_in_array[1])

    with pytest.raises(
        TypeError,
        match=r"words argument should be a list, tuple or np.array of strings, got .*",
    ):
        word2vec_sm.update_embeddings(None, embeddings)

    with pytest.raises(
        TypeError, match=r"embeddings should be a list, tuple or np.array, got:.*",
    ):
        word2vec_sm.update_embeddings(words, None)

    with pytest.raises(
        ValueError, match=r"words and embeddings must have the same size, got:.*",
    ):
        word2vec_sm.update_embeddings(words + ["is"], embeddings)


# -------------------------------------------------------------------------------------
def test_get_embeddings_from_word_set(word2vec_sm):

    WORDS = ["man", "woman"]

    with pytest.raises(
        TypeError,
        match=r"word_set should be a list, tuple, set or np.array of strings, got.*",
    ):
        word2vec_sm.get_embeddings_from_word_set(None, preprocessor_args=1)

    with pytest.raises(
        TypeError,
        match="preprocessor_args should be a dict of preprocessor arguments, got",
    ):
        word2vec_sm.get_embeddings_from_word_set(WORDS, preprocessor_args=1)

    with pytest.raises(
        TypeError,
        match="secondary_preprocessor_args should be a dict of preprocessor arguments or "
        "None, got",
    ):
        word2vec_sm.get_embeddings_from_word_set(WORDS, secondary_preprocessor_args=-1)

    # ----------------------------------------------------------------------------------
    # test basic opretaion of _get_embeddings_from_word_set
    WORDS = ["man", "woman"]

    not_found_words, embeddings = word2vec_sm.get_embeddings_from_word_set(WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(word2vec_sm["man"], embeddings["man"])
    assert np.array_equal(word2vec_sm["woman"], embeddings["woman"])

    # test with a word that does not exists in the model
    WORDS = ["man", "woman", "not_a_word_"]
    not_found_words, embeddings = word2vec_sm.get_embeddings_from_word_set(WORDS)

    assert len(embeddings) == 2
    assert len(not_found_words) == 1

    assert list(embeddings.keys()) == ["man", "woman"]
    assert ["not_a_word_"] == not_found_words

    assert np.array_equal(word2vec_sm["man"], embeddings["man"])
    assert np.array_equal(word2vec_sm["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test word preprocessor lowercase
    WORDS = [
        "mAN",
        "WOmaN",
    ]
    not_found_words, embeddings = word2vec_sm.get_embeddings_from_word_set(
        WORDS, {"lowercase": True}
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(word2vec_sm["man"], embeddings["man"])
    assert np.array_equal(word2vec_sm["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # test word preprocessor strip_accents:
    WORDS = [
        "mán",
        "wömàn",
    ]
    not_found_words, embeddings = word2vec_sm.get_embeddings_from_word_set(
        WORDS, {"strip_accents": True}
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 0

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == []

    assert np.array_equal(word2vec_sm["man"], embeddings["man"])
    assert np.array_equal(word2vec_sm["woman"], embeddings["woman"])

    # ----------------------------------------------------------------------------------
    # secondary_preprocessor_options:
    WORDS = ["mán", "wömàn", "qwerty", "ásdf"]
    not_found_words, embeddings = word2vec_sm.get_embeddings_from_word_set(
        WORDS, secondary_preprocessor_args={"strip_accents": True}
    )

    assert len(embeddings) == 2
    assert len(not_found_words) == 2

    assert list(embeddings.keys()) == ["man", "woman"]
    assert not_found_words == ["qwerty", "ásdf"]

    assert np.array_equal(word2vec_sm["man"], embeddings["man"])
    assert np.array_equal(word2vec_sm["woman"], embeddings["woman"])


# -------------------------------------------------------------------------------------
def test_get_embeddings_from_pairs(word2vec_sm, caplog):

    with pytest.raises(
        TypeError,
        match=(
            r"pairs should be a list, tuple, set or np.array of pairs of strings"
            r", got:.*"
        ),
    ):
        word2vec_sm.get_embeddings_from_pairs(None)

    with pytest.raises(
        TypeError,
        match=(
            r"Every pair in pairs must be a list, set, tuple or np.array of strings"
            r", got in index.*"
        ),
    ):
        word2vec_sm.get_embeddings_from_pairs([None])

    with pytest.raises(
        ValueError, match=r"Every pair should have length 2. Got in index.*",
    ):
        word2vec_sm.get_embeddings_from_pairs([["she", "he", "it"]])

    with pytest.raises(
        TypeError,
        match=(
            r"All elements of a pair should be strings. "
            r"Got in index.*at position 0:.*"
        ),
    ):
        word2vec_sm.get_embeddings_from_pairs([[1, "he"]])

    with pytest.raises(
        TypeError,
        match=(
            r"All elements of a pair should be strings. "
            r"Got in index.* at position 1:.*"
        ),
    ):
        word2vec_sm.get_embeddings_from_pairs([["she", 1]])

    with pytest.raises(
        TypeError, match=r"pairs_set_name should be a string or None, got:.*",
    ):
        word2vec_sm.get_embeddings_from_pairs([["she", "he"]], 0)

    with pytest.raises(
        TypeError, match=r"warn_lost_pairs should be a bool, got:.*",
    ):
        word2vec_sm.get_embeddings_from_pairs([["she", "he"]], "definning", None)

    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got:.*",
    ):
        word2vec_sm.get_embeddings_from_pairs([["she", "he"]], "definning", True, None)

    pairs = [["woman", "man"], ["she", "he"], ["mother", "father"]]
    pairs_set_name = "definning"

    embedding_pairs = word2vec_sm.get_embeddings_from_pairs(
        pairs=pairs, pairs_set_name=pairs_set_name
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
            assert all(word2vec_sm[w] == e)

    oov_pairs = [["the", "vbbge"], ["ddsds", "ferhh"]]
    pairs_with_oov = pairs + oov_pairs

    with caplog.at_level(logging.INFO):
        embedding_pairs_2 = word2vec_sm.get_embeddings_from_pairs(
            pairs=pairs_with_oov,
            pairs_set_name=pairs_set_name,
            warn_lost_pairs=True,
            verbose=True,
        )
        assert len(embedding_pairs_2) == 3
        assert (
            "The word(s) ['vbbge'] of the definning pair at index 3 were not found. "
            "This pair will be omitted." in caplog.text
        )
        assert "The word(s) ['ddsds', 'ferhh'] of the definning pair at index 4 were "
        "not found. This pair will be omitted." in caplog.text
        pass

        assert (
            "3/5 pairs of words were correctly converted to pairs of embeddings"
            in caplog.text
        )

    with pytest.raises(
        Exception,
        match=r"No pair could be converted to embedding because no pair "
        "could be fully found in the model vocabulary.",
    ):
        word2vec_sm.get_embeddings_from_pairs(pairs=oov_pairs)


def test_warn_not_found_words(word2vec_sm, caplog):

    word2vec_sm._warn_not_found_words("Set1", ["aaa", "bbb"])
    msg = "The following words from set 'Set1' do not exist within the vocabulary"
    assert msg in caplog.text
    assert "['aaa', 'bbb']" in caplog.text


@pytest.fixture
def simple_model_and_query():
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, "weat_w2v", "")
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
    return w2v, model, query, flowers, insects, pleasant, unpleasant


def test_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # test types

    # target sets None
    with pytest.raises(TypeError, match="query should be an instance of Query, got"):
        model.get_embeddings_from_query(None)

    # target sets int
    with pytest.raises(
        TypeError, match="lost_vocabulary_threshold should be float or np.floating, got"
    ):
        model.get_embeddings_from_query(query, lost_vocabulary_threshold=None)

    with pytest.raises(
        TypeError,
        match="preprocessor_args should be a dict of preprocessor arguments, got",
    ):
        model.get_embeddings_from_query(query, preprocessor_args=1)

    with pytest.raises(
        TypeError,
        match=(
            "secondary_preprocessor_args should be a dict of preprocessor arguments"
            " or None, got"
        ),
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


def test_preprocessor_args_on_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # with lost words and warn_not_found_words=True
    flowers_2 = flowers + ["aaa", "bbb"]
    query_2 = Query(
        [flowers_2, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = model.get_embeddings_from_query(query_2, warn_not_found_words=True)
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
    embeddings = model.get_embeddings_from_query(
        query_3, preprocessor_args={"lowercase": True}
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
    embeddings = model.get_embeddings_from_query(
        query_3, secondary_preprocessor_args={"lowercase": True}
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


def test_threshold_param_on_get_embeddings_from_query(caplog, simple_model_and_query):
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    # with lost vocabulary theshold.
    flowers_ = flowers + ["aaa", "aab", "aac", "aad", "aaf", "aag", "aah", "aai", "aaj"]
    query = Query(
        [flowers_, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Flowers' into" in caplog.text

    # with lost vocabulary theshold.
    insects_ = insects + ["aaa", "aab", "aac", "aad", "aaf", "aag", "aah", "aai", "aaj"]
    query = Query(
        [flowers, insects_],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Insects' into" in caplog.text

    # with lost vocabulary theshold.
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
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Pleasant' into" in caplog.text

    # test attribute 2 with lost vocabulary theshold.
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
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.1)

    assert embeddings is None
    assert "The transformation of 'Unpleasant' into" in caplog.text

    # with lost vocabulary theshold.
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
    embeddings = model.get_embeddings_from_query(query, lost_vocabulary_threshold=0.5)

    assert embeddings is not None
