"""Metrics Testing"""
import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors
from wefe.datasets.datasets import load_weat
from wefe.metrics import ECT, MAC, RIPA, RND, RNSB, WEAT
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
def weat_wordsets():
    return load_weat()


def test_WEAT(model, weat_wordsets):

    weat = WEAT()
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    results = weat.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
    assert isinstance(results["result"], np.number)
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)
    assert results["result"] == results["weat"]
    assert np.isnan(results["p_value"])

    results = weat.run_query(query, model, return_effect_size=True)
    assert isinstance(results["result"], np.number)
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)
    assert results["result"] == results["effect_size"]
    assert np.isnan(results["p_value"])

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="left-sided",
    )

    assert isinstance(results["result"], np.number)
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)
    assert isinstance(results["p_value"], (float, np.number))

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="right-sided",
    )

    assert isinstance(results["result"], np.number)
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)
    assert isinstance(results["p_value"], (float, np.number))

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="two-sided",
    )

    assert isinstance(results["result"], np.number)
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)
    assert isinstance(results["p_value"], (float, np.number))


def test_RND(model, weat_wordsets):

    rnd = RND()
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    # test with euclidean distance
    results = rnd.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], np.number)
    assert isinstance(results["rnd"], np.number)
    assert isinstance(results["distances_by_word"], dict)
    assert len(results["distances_by_word"]) > 0

    # test with cosine distance
    results = rnd.run_query(query, model, distance="cos")

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], np.number)
    assert isinstance(results["rnd"], np.number)
    assert isinstance(results["distances_by_word"], dict)
    assert len(results["distances_by_word"]) > 0

    with pytest.raises(
        ValueError, match=r'distance_type can be either "norm" or "cos", .*'
    ):
        rnd.run_query(query, model, distance="other_distance")

    # lost word threshold test
    results = rnd.run_query(
        Query(
            [["bla", "asd"], weat_wordsets["insects"]],
            [weat_wordsets["pleasant_5"]],
            ["Flowers", "Insects"],
            ["Pleasant"],
        ),
        model,
    )
    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert np.isnan(results["result"])
    assert np.isnan(results["rnd"])
    assert isinstance(results["distances_by_word"], dict)
    assert len(results["distances_by_word"]) == 0


def test_RNSB(capsys, model, weat_wordsets):

    rnsb = RNSB()
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    results = rnsb.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
    assert list(results.keys()) == [
        "query_name",
        "result",
        "rnsb",
        "negative_sentiment_probabilities",
        "negative_sentiment_distribution",
    ]
    assert isinstance(results["result"], (np.float32, np.float64, float, np.float_))
    assert isinstance(results["negative_sentiment_probabilities"], dict)
    assert isinstance(results["negative_sentiment_distribution"], dict)

    query = Query(
        [
            weat_wordsets["flowers"],
            weat_wordsets["instruments"],
            weat_wordsets["male_terms"],
            weat_wordsets["female_terms"],
        ],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects", "Male terms", "Female terms"],
        ["Pleasant", "Unpleasant"],
    )
    results = rnsb.run_query(query, model)

    assert (
        results["query_name"]
        == "Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant"
    )
    assert isinstance(results["result"], np.number)

    # custom classifier, print model eval
    results = rnsb.run_query(query, model, print_model_evaluation=True)

    print(capsys.readouterr())
    captured = capsys.readouterr()
    assert "Classification Report" in captured.out

    assert (
        results["query_name"]
        == "Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant"
    )
    assert isinstance(results["result"], np.number)
    assert isinstance(results["rnsb"], np.number)

    # lost word threshold test
    results = rnsb.run_query(
        Query(
            [["bla", "asd"], weat_wordsets["insects"]],
            [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
            ["Flowers", "Insects"],
            ["Pleasant", "Unpleasant"],
        ),
        model,
    )
    assert np.isnan(results["rnsb"])
    assert np.isnan(results["result"])
    assert isinstance(results["negative_sentiment_probabilities"], dict)
    assert isinstance(results["negative_sentiment_distribution"], dict)
    assert len(results["negative_sentiment_probabilities"]) == 0
    assert len(results["negative_sentiment_distribution"]) == 0

    # test random state
    query = Query(
        [
            weat_wordsets["flowers"],
            weat_wordsets["instruments"],
            weat_wordsets["male_terms"],
            weat_wordsets["female_terms"],
        ],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects", "Male terms", "Female terms"],
        ["Pleasant", "Unpleasant"],
    )
    results = rnsb.run_query(query, model, random_state=42)

    assert (
        results["query_name"]
        == "Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant"
    )
    assert isinstance(results["result"], np.number)


def test_MAC(model, weat_wordsets):

    mac = MAC()
    query = Query(
        [weat_wordsets["flowers"]],
        [
            weat_wordsets["pleasant_5"],
            weat_wordsets["pleasant_9"],
            weat_wordsets["unpleasant_5"],
            weat_wordsets["unpleasant_9"],
        ],
        ["Flowers"],
        ["Pleasant 5 ", "Pleasant 9", "Unpleasant 5", "Unpleasant 9"],
    )
    results = mac.run_query(query, model)

    assert (
        results["query_name"]
        == "Flowers wrt Pleasant 5 , Pleasant 9, Unpleasant 5 and Unpleasant 9"
    )
    assert isinstance(results["result"], np.number)
    assert isinstance(results["mac"], np.number)
    assert isinstance(results["targets_eval"], dict)
    assert len(results["targets_eval"]["Flowers"]) == len(weat_wordsets["flowers"])
    # 4 = number of attribute sets
    assert len(results["targets_eval"]["Flowers"][weat_wordsets["flowers"][0]]) == 4

    # test metric with a target set that loses more words than allowed.
    query = Query(
        [weat_wordsets["flowers"], ["blabla", "asdf"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = mac.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert np.isnan(results["mac"])
    assert np.isnan(results["result"])


def test_ECT(model, weat_wordsets):

    ect = ECT()
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = ect.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], np.number)
    assert isinstance(results["ect"], np.number)

    # test metric with a target set that loses more words than allowed.
    query = Query(
        [weat_wordsets["flowers"], ["blabla", "asdf"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = ect.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert np.isnan(results["ect"])
    assert np.isnan(results["result"])


def test_RIPA(model, weat_wordsets):

    ripa = RIPA()
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = ripa.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["ripa"], (np.float32, np.float64, float))
    assert isinstance(results["word_values"], dict)

    for word, word_value in results["word_values"].items():
        assert isinstance(word, str)
        assert isinstance(word_value, dict)
        assert isinstance(word_value["mean"], (np.float32, np.float64, float))
        assert isinstance(word_value["std"], (np.float32, np.float64, float))
