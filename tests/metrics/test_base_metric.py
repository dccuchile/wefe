import pytest

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_base_metric_input_checking(
    model: WordEmbeddingModel,
    query_2t2a_1: Query,
    query_3t2a_1: Query,
):
    # Create and configure base metric testing.
    # disable abstract methods.

    # instance test metric
    BaseMetric.__abstractmethods__ = frozenset()
    base_metric = BaseMetric()
    base_metric.metric_template = (2, 3)
    base_metric.metric_name = "Test Metric"
    base_metric.metric_short_name = "TM"

    with pytest.raises(TypeError, match="query should be a Query instance, got*"):
        base_metric._check_input(None, model, {})

    with pytest.raises(
        TypeError, match="word_embedding should be a WordEmbeddingModel instance, got*"
    ):
        base_metric._check_input(query_2t2a_1, None, {})

    with pytest.raises(
        Exception,
        match="The cardinality of the set of target words of the 'Flowers, Weapons and "
        "Instruments wrt Pleasant and Unpleasant' query does not match with the "
        "cardinality required by TM. Provided query: 3, metric: 2",
    ):
        base_metric._check_input(query_3t2a_1, model, {})

    with pytest.raises(
        Exception,
        match=(
            "The cardinality of the set of attribute words of the 'Flowers and Insects "
            "wrt Pleasant and Unpleasant' query does not match with the cardinality "
            "required by TM. Provided query: 2, metric: 3"
        ),
    ):
        base_metric._check_input(query_2t2a_1, model, {})


def test_validate_old_preprocessor_args_inputs(
    model: WordEmbeddingModel,
    query_2t2a_1: Query,
):
    # instance test metric
    BaseMetric.__abstractmethods__ = frozenset()
    base_metric = BaseMetric()
    base_metric.metric_template = (2, 2)
    base_metric.metric_name = "Test Metric"
    base_metric.metric_short_name = "TM"

    with pytest.raises(
        DeprecationWarning,
        match=(
            r"preprocessor_args argument is deprecated. "
            r"Use preprocessors=\[\{'uppercase': True\}\] instead.*."
        ),
    ):
        base_metric._check_input(
            query_2t2a_1, model, {"preprocessor_args": {"uppercase": True}}
        )

    with pytest.raises(
        DeprecationWarning,
        match=(
            r"secondary_preprocessor_args is deprecated. "
            r"Use preprocessors=\[\{\}, \{'uppercase': True\}\] instead.*."
        ),
    ):
        base_metric._check_input(
            query_2t2a_1, model, {"secondary_preprocessor_args": {"uppercase": True}}
        )

    with pytest.raises(
        DeprecationWarning,
        match=(
            r"preprocessor_args and secondary_preprocessor_args arguments are "
            r"deprecated. Use preprocessors=\[\{'uppercase': True\}, \{'uppercase': "
            r"True\}\] instead."
        ),
    ):
        base_metric._check_input(
            query_2t2a_1,
            model,
            {
                "preprocessor_args": {"uppercase": True},
                "secondary_preprocessor_args": {"uppercase": True},
            },
        )


def test_run_query(model: WordEmbeddingModel, query_2t2a_1: Query):
    # disable abstract methods.
    BaseMetric.__abstractmethods__ = frozenset()
    base_metric = BaseMetric()

    base_metric.metric_template = (2, 2)
    base_metric.metric_name = "Test Metric"
    base_metric.metric_short_name = "TM"

    with pytest.raises(
        NotImplementedError,
    ):
        base_metric.run_query(query_2t2a_1, model)
