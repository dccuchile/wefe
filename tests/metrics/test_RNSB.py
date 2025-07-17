"""RNSB metric testing."""

from typing import Any

import numpy as np

from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_RNSB_result_keys(results: dict[str, Any]) -> None:
    assert list(results.keys()) == [
        "query_name",
        "result",
        "rnsb",
        "negative_sentiment_probabilities",
        "negative_sentiment_distribution",
    ]


def check_RNSB_result_values(results: dict[str, Any]) -> None:
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    # check result type and probability interval
    assert isinstance(results["result"], np.number)
    assert 0 <= results["result"] <= 1

    # check negative_sentiment_probabilities
    negative_sentiment_probabilities = results["negative_sentiment_probabilities"]
    assert isinstance(negative_sentiment_probabilities, dict)
    assert len(negative_sentiment_probabilities) > 0
    for word, proba in negative_sentiment_probabilities.items():
        assert isinstance(word, str)
        assert isinstance(proba, float)
        assert 0 <= proba <= 1

    # check negative_sentiment_distribution
    negative_sentiment_distribution = results["negative_sentiment_distribution"]
    assert isinstance(negative_sentiment_distribution, dict)
    assert len(negative_sentiment_distribution) > 0

    for word, proba in negative_sentiment_distribution.items():
        assert isinstance(word, str)
        assert isinstance(proba, float)
        assert 0 <= proba <= 1

    assert len(negative_sentiment_probabilities) == len(negative_sentiment_distribution)


def test_RNSB_base(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    rnsb = RNSB()
    results = rnsb.run_query(query_2t2a_1, model)
    check_RNSB_result_keys(results)
    check_RNSB_result_values(results)
    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"


def test_RNSB_more_targets(model: WordEmbeddingModel, query_4t2a_1: Query) -> None:
    rnsb = RNSB()
    results = rnsb.run_query(query_4t2a_1, model)
    check_RNSB_result_keys(results)
    check_RNSB_result_values(results)
    assert (
        results["query_name"]
        == "Flowers, Insects, Instruments and Weapons wrt Pleasant and Unpleasant"
    )


def test_RNSB_print_model_evaluation(
    capsys, model: WordEmbeddingModel, query_2t2a_1: Query
) -> None:
    rnsb = RNSB()
    results = rnsb.run_query(query_2t2a_1, model, print_model_evaluation=True)
    check_RNSB_result_keys(results)
    check_RNSB_result_values(results)

    print(capsys.readouterr())
    captured = capsys.readouterr()
    assert "Classification Report" in captured.out

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"


def test_RNSB_no_holdout(capsys, model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    rnsb = RNSB()
    results = rnsb.run_query(
        query_2t2a_1, model, holdout=False, print_model_evaluation=True
    )
    check_RNSB_result_keys(results)
    check_RNSB_result_values(results)

    print(capsys.readouterr())
    captured = capsys.readouterr()
    assert "Holdout is disabled. No evaluation was performed" in captured.out

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"


def test_RNSB_lost_vocabulary_threshold(
    model: WordEmbeddingModel, query_2t2a_lost_vocab_1: Query
) -> None:
    rnsb = RNSB()
    results = rnsb.run_query(query_2t2a_lost_vocab_1, model)

    check_RNSB_result_keys(results)

    assert np.isnan(results["rnsb"])
    assert np.isnan(results["result"])
    assert isinstance(results["negative_sentiment_probabilities"], dict)
    assert isinstance(results["negative_sentiment_distribution"], dict)
    assert len(results["negative_sentiment_probabilities"].keys()) == 0
    assert len(results["negative_sentiment_distribution"].keys()) == 0

    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"


def test_RNSB_with_random_state(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    rnsb = RNSB()
    results_1 = rnsb.run_query(query_2t2a_1, model, random_state=42)
    check_RNSB_result_keys(results_1)
    check_RNSB_result_values(results_1)

    results_2 = rnsb.run_query(query_2t2a_1, model, random_state=42)
    check_RNSB_result_keys(results_2)
    check_RNSB_result_values(results_2)

    assert results_1 == results_2
