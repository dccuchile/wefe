"""ECT metric testing."""

from typing import Any

import numpy as np

from wefe.metrics import ECT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_ECT_result_keys(results: dict[str, Any]):
    assert list(results.keys()) == [
        "query_name",
        "result",
        "ect",
    ]


def check_ECT_result_values(results: dict[str, Any]):
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    # check result type
    assert isinstance(results["result"], np.number)
    assert isinstance(results["ect"], np.number)
    assert -1 <= results["ect"] <= 1


def test_ECT(model: WordEmbeddingModel, query_2t1a_1: Query):
    ect = ECT()
    results = ect.run_query(query_2t1a_1, model)

    check_ECT_result_keys(results)
    check_ECT_result_values(results)
    assert results["query_name"] == "Flowers and Insects wrt Pleasant"


def test_ECT_lost_vocabulary_threshold(
    model: WordEmbeddingModel, query_2t1a_lost_vocab_1: Query
):
    # test metric with a target set that loses more words than allowed.
    ect = ECT()
    results = ect.run_query(query_2t1a_lost_vocab_1, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert np.isnan(results["ect"])
    assert np.isnan(results["result"])
