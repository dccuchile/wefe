"""WEAT metric testing."""

from typing import Any

import numpy as np

from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_WEAT_result_keys(results: dict[str, Any]) -> None:
    assert list(results.keys()) == [
        "query_name",
        "result",
        "weat",
        "effect_size",
        "p_value",
    ]


def check_WEAT_result_values(results: dict[str, Any]) -> None:
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    # check result type
    assert isinstance(results["result"], np.number)

    # check metrics type
    assert isinstance(results["weat"], np.number)
    assert isinstance(results["effect_size"], np.number)

    # check p_value options
    assert isinstance(results["p_value"], (float, np.number)) or np.isnan(
        results["p_value"]
    )


def test_WEAT(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    weat = WEAT()

    results = weat.run_query(query_2t2a_1, model)

    check_WEAT_result_keys(results)
    check_WEAT_result_values(results)
    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
    assert results["result"] == results["weat"]


def test_WEAT_effect_size(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    weat = WEAT()

    results = weat.run_query(query_2t2a_1, model, return_effect_size=True)
    check_WEAT_result_keys(results)
    check_WEAT_result_values(results)

    assert results["result"] == results["effect_size"]


def test_WEAT_left_sided_p_value(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    weat = WEAT()

    results = weat.run_query(
        query_2t2a_1,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="left-sided",
    )
    check_WEAT_result_keys(results)
    check_WEAT_result_values(results)
    assert isinstance(results["p_value"], float)


def test_WEAT_right_sided_p_value(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    weat = WEAT()

    results = weat.run_query(
        query_2t2a_1,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="right-sided",
    )

    check_WEAT_result_keys(results)
    check_WEAT_result_values(results)
    assert isinstance(results["p_value"], float)


def test_WEAT_two_sided_p_value(model: WordEmbeddingModel, query_2t2a_1: Query) -> None:
    weat = WEAT()

    results = weat.run_query(
        query_2t2a_1,
        model,
        calculate_p_value=True,
        p_value_iterations=100,
        p_value_test_type="two-sided",
    )

    check_WEAT_result_keys(results)
    check_WEAT_result_values(results)
    assert isinstance(results["p_value"], float)
