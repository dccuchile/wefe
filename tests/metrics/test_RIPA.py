"""RIPA metric testing."""

from typing import Any

import numpy as np

from wefe.metrics import RIPA
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_RIPA_result_keys(results: dict[str, Any]) -> None:
    assert list(results.keys()) == ["query_name", "result", "ripa", "word_values"]


def check_RIPA_result_values(results: dict[str, Any]) -> None:
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    assert isinstance(results["result"], np.number | float)
    assert isinstance(results["ripa"], np.number | float)
    assert isinstance(results["word_values"], dict)

    for word, word_value in results["word_values"].items():
        assert isinstance(word, str)
        assert isinstance(word_value, dict)
        assert isinstance(word_value["mean"], np.number | float)
        assert isinstance(word_value["std"], np.number | float)


def test_RIPA(model: WordEmbeddingModel, query_2t1a_1: Query) -> None:
    ripa = RIPA()

    results = ripa.run_query(query_2t1a_1, model)

    check_RIPA_result_keys(results)
    check_RIPA_result_values(results)
    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
