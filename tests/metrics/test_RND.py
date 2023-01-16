"""RND metric testing"""
from typing import Any, Dict

import numpy as np
import pytest

from wefe.metrics import RND
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_RND_result_keys(results: Dict[str, Any]):
    assert list(results.keys()) == [
        "query_name",
        "result",
        "rnd",
        "distance_by_word",
    ]


def check_RND_result_values(results: Dict[str, Any]):
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    # check result type
    assert isinstance(results["result"], np.number)
    assert isinstance(results["rnd"], np.number)

    distances_by_word = results["distance_by_word"]
    assert isinstance(distances_by_word, dict)
    assert len(distances_by_word) > 0
    for word, distance in distances_by_word.items():
        assert isinstance(word, str)
        assert isinstance(distance, (float, np.number))
        assert len(word) > 0


def test_RND_with_euclidean_distance(model: WordEmbeddingModel, query_2t1a_1: Query):
    # note: the euclidean distance is the default distance.
    rnd = RND()
    result = rnd.run_query(query_2t1a_1, model)

    check_RND_result_keys(result)
    check_RND_result_values(result)
    assert result["query_name"] == "Flowers and Insects wrt Pleasant"


def test_RND_with_cosine_distance(model: WordEmbeddingModel, query_2t1a_1: Query):
    rnd = RND()
    result = rnd.run_query(query_2t1a_1, model, distance="cos")

    check_RND_result_keys(result)
    check_RND_result_values(result)
    assert result["query_name"] == "Flowers and Insects wrt Pleasant"


def test_RND_wrong_distance_type_parameter(
    model: WordEmbeddingModel, query_2t1a_1: Query
):
    rnd = RND()

    with pytest.raises(
        ValueError, match=r'distance_type can be either "norm" or "cos", .*'
    ):
        rnd.run_query(query_2t1a_1, model, distance="other_distance")


def test_RND_lost_vocabulary_threshold(
    model: WordEmbeddingModel, query_2t1a_lost_vocab_1: Query
):
    rnd = RND()

    result = rnd.run_query(
        query_2t1a_lost_vocab_1,
        model,
    )
    check_RND_result_keys(result)

    assert result["query_name"] == "Flowers and Insects wrt Pleasant"

    assert np.isnan(result["result"])
    assert np.isnan(result["rnd"])

    assert isinstance(result["distance_by_word"], dict)
    assert len(result["distance_by_word"]) == 0
