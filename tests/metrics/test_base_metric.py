"""Unit tests for the BaseMetric class in the wefe.metrics.base_metric module."""

import pytest

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_base_metric_input_checking(
    model: WordEmbeddingModel,
    query_2t2a_1: Query,
    query_3t2a_1: Query,
) -> None:
    """Test input validation for the BaseMetric class.
    This test verifies that the `_check_input` method of `BaseMetric` correctly
    raises exceptions when provided with invalid inputs, such as incorrect types
    for `query` and `model`, or mismatched cardinalities for target and attribute sets.

    Parameters
    ----------
    model : WordEmbeddingModel
        A word embedding model instance used for metric evaluation.
    query_2t2a_1 : Query
        A query instance with 2 target sets and 2 attribute sets.
    query_3t2a_1 : Query
        A query instance with 3 target sets and 2 attribute sets.

    """
    # Create and configure base metric testing.
    # disable abstract methods.

    # instance test metric
    BaseMetric.__abstractmethods__ = frozenset()
    base_metric = BaseMetric()
    base_metric.metric_template = (2, 3)
    base_metric.metric_name = "Test Metric"
    base_metric.metric_short_name = "TM"

    with pytest.raises(TypeError, match="query should be a Query instance, got*"):
        base_metric._check_input(
            query=None,
            model=model,
            lost_vocabulary_threshold=0.2,
            warn_not_found_words=True,
        )

    with pytest.raises(
        TypeError,
        match="model should be a WordEmbeddingModel instance, got: <class 'NoneType'>.",
    ):
        base_metric._check_input(
            query=query_2t2a_1,
            model=None,
            lost_vocabulary_threshold=0.2,
            warn_not_found_words=True,
        )

    with pytest.raises(
        Exception,
        match=(
            r"The cardinality of the target sets of the 'Flowers, Weapons and "
            r"Instruments wrt Pleasant and Unpleasant' query \(3\) does not match "
            r"the cardinality required by TM \(2\)."
        ),
    ):
        base_metric._check_input(
            query=query_3t2a_1,
            model=model,
            lost_vocabulary_threshold=0.2,
            warn_not_found_words=True,
        )

    with pytest.raises(
        Exception,
        match=(
            r"The cardinality of the attribute sets of the 'Flowers and Insects wrt "
            r"Pleasant and Unpleasant' query \(2\) does not match the cardinality "
            r"required by TM \(3\)."
        ),
    ):
        base_metric._check_input(
            query=query_2t2a_1,
            model=model,
            lost_vocabulary_threshold=0.2,
            warn_not_found_words=True,
        )


def test_run_query(
    model: WordEmbeddingModel,
    query_2t2a_1: Query,
) -> None:
    """Test that the `run_query` method of `BaseMetric` raises a NotImplementedError.

    Parameters
    ----------
    model : WordEmbeddingModel
        The word embedding model to be used in the query.
    query_2t2a_1 : Query
        The query object to be passed to the metric.

    Raises
    ------
    NotImplementedError
        If the `run_query` method is not implemented in `BaseMetric`.

    """
    # disable abstract methods.
    BaseMetric.__abstractmethods__ = frozenset()
    base_metric = BaseMetric()

    base_metric.metric_template = (2, 2)
    base_metric.metric_name = "Test Metric"
    base_metric.metric_short_name = "TM"

    with pytest.raises(
        NotImplementedError,
    ):
        base_metric.run_query(query=query_2t2a_1, model=model)
