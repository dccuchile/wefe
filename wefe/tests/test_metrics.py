import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors

from wefe.word_embedding_model import WordEmbeddingModel
from wefe.datasets.datasets import load_weat
from wefe.query import Query
from wefe.metrics import WEAT, RND, RNSB, MAC, ECT


@pytest.fixture
def model():
    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")
    return WordEmbeddingModel(w2v, "word2vec")


@pytest.fixture
def weat_word_set():
    return load_weat()


def test_WEAT(model, weat_word_set):

    weat = WEAT()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
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


def test_RND(model, weat_word_set):

    rnd = RND()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = rnd.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], np.number)


def test_RNSB(capsys, model, weat_word_set):

    rnsb = RNSB()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    results = rnsb.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
    assert list(results.keys()) == [
        "query_name",
        "result",
        "kl-divergence",
        "clf_accuracy",
        "negative_sentiment_probabilities",
        "negative_sentiment_distribution",
    ]
    assert isinstance(results["result"], (np.float32, np.float64, float, np.float_))
    assert isinstance(results["negative_sentiment_probabilities"], dict)
    assert isinstance(results["negative_sentiment_distribution"], dict)

    query = Query(
        [
            weat_word_set["flowers"],
            weat_word_set["instruments"],
            weat_word_set["male_terms"],
            weat_word_set["female_terms"],
        ],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
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

    # lost word threshold test
    results = rnsb.run_query(
        Query(
            [["bla", "asd"], weat_word_set["insects"]],
            [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
            ["Flowers", "Insects"],
            ["Pleasant", "Unpleasant"],
        ),
        model,
    )
    assert np.isnan(np.nan)


def test_MAC(model, weat_word_set):

    mac = MAC()
    query = Query(
        [weat_word_set["flowers"]],
        [
            weat_word_set["pleasant_5"],
            weat_word_set["pleasant_9"],
            weat_word_set["unpleasant_5"],
            weat_word_set["unpleasant_9"],
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


def test_ECT(model, weat_word_set):

    ect = ECT()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = ect.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], np.number)
