import logging
import pytest

import numpy as np

from ..utils import load_weat_w2v
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query
from ..metrics import WEAT, RND, RNSB, MAC, ECT, RIPA

LOGGER = logging.getLogger(__name__)


def test_WEAT():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

    weat = WEAT()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    results = weat.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["weat"], (np.float32, np.float64, float))
    assert isinstance(results["effect_size"], (np.float32, np.float64, float))
    assert results["result"] == results["weat"]
    assert np.isnan(results["p_value"])

    results = weat.run_query(query, model, return_effect_size=True)
    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["weat"], (np.float32, np.float64, float))
    assert isinstance(results["effect_size"], (np.float32, np.float64, float))
    assert results["result"] == results["effect_size"]
    assert np.isnan(results["p_value"])

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="left-sided",
    )

    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["weat"], (np.float32, np.float64, float))
    assert isinstance(results["effect_size"], (np.float32, np.float64, float))
    assert isinstance(results["p_value"], (np.float32, np.float64, float))

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="right-sided",
    )

    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["weat"], (np.float32, np.float64, float))
    assert isinstance(results["effect_size"], (np.float32, np.float64, float))
    assert isinstance(results["p_value"], (np.float32, np.float64, float))

    results = weat.run_query(
        query,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="two-sided",
    )

    assert isinstance(results["result"], (np.float32, np.float64, float))
    assert isinstance(results["weat"], (np.float32, np.float64, float))
    assert isinstance(results["effect_size"], (np.float32, np.float64, float))
    assert isinstance(results["p_value"], (np.float32, np.float64, float))


def test_RND():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

    rnd = RND()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = rnd.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], (np.float32, np.float64, float))


def test_RNSB(capsys):

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

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
    assert isinstance(results["result"], (np.float32, np.float64, float))

    # custom classifier, print model eval
    results = rnsb.run_query(query, model, print_model_evaluation=True)

    print(capsys.readouterr())
    captured = capsys.readouterr()
    assert "Classification Report" in captured.out

    assert (
        results["query_name"]
        == "Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant"
    )
    assert isinstance(results["result"], (np.float32, np.float64, float))

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


def test_MAC():
    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

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
    assert isinstance(results["result"], (np.float32, np.float64, float))


def test_ECT():
    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

    ect = ECT()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    results = ect.run_query(query, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], (np.float32, np.float64, float))


def test_RIPA():
    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), "weat_w2v", "")

    ripa = RIPA()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
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

