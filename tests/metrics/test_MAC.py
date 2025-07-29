"""MAC metric testing."""

from typing import Any

import numpy as np

from wefe.metrics import MAC
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_MAC_result_keys(results: dict[str, Any]) -> None:
    assert list(results.keys()) == ["query_name", "result", "mac", "targets_eval"]


def check_MAC_result_values(results: dict[str, Any]) -> None:
    # note: this checking only applies when the result is not np.nan.
    assert isinstance(results["query_name"], str)

    # check result type
    assert isinstance(results["result"], np.number)

    # check metrics type
    assert isinstance(results["mac"], np.number)

    targets_eval = results["targets_eval"]
    assert isinstance(targets_eval, dict)

    for target_name, target_eval in targets_eval.items():
        assert isinstance(target_name, str)
        assert isinstance(target_eval, dict)
        for target_word, attribute_scores in target_eval.items():
            assert isinstance(target_word, str)
            assert isinstance(attribute_scores, dict)
            for attribute_name, attribute_score in attribute_scores.items():
                assert isinstance(attribute_name, str)
                assert isinstance(attribute_score, np.number | float)


def test_MAC(model, query_1t4_1) -> None:
    mac = MAC()
    results = mac.run_query(query_1t4_1, model)

    assert (
        results["query_name"]
        == "Flowers wrt Pleasant 5 , Pleasant 9, Unpleasant 5 and Unpleasant 9"
    )

    check_MAC_result_keys(results)
    check_MAC_result_values(results)

    assert len(results["targets_eval"]["Flowers"]) == len(query_1t4_1.target_sets[0])
    # 4 = number of attribute sets
    for word in query_1t4_1.target_sets[0]:
        assert len(results["targets_eval"]["Flowers"][word]) == 4


def test_MAC_lost_vocabulary_threshold(
    model: WordEmbeddingModel, query_2t2a_lost_vocab_1: Query
) -> None:
    mac = MAC()

    # test metric with a target set that loses more words than allowed.
    results = mac.run_query(query_2t2a_lost_vocab_1, model)

    check_MAC_result_keys(results)
    assert np.isnan(results["mac"])
    assert np.isnan(results["result"])
    assert isinstance(results["targets_eval"], dict)
    assert results["query_name"] == "Flowers and Insects wrt Pleasant and Unpleasant"
