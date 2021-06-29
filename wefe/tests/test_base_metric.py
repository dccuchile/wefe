import pytest
from wefe.utils import load_test_model
from wefe.metrics.base_metric import BaseMetric
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.datasets.datasets import load_weat
from wefe.query import Query


@pytest.fixture
def simple_model_and_query():
    test_model = load_test_model()
    weat_wordsets = load_weat()

    flowers = weat_wordsets["flowers"]
    insects = weat_wordsets["insects"]
    pleasant = weat_wordsets["pleasant_5"]
    unpleasant = weat_wordsets["unpleasant_5"]
    query = Query(
        [flowers, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    return test_model, query, flowers, insects, pleasant, unpleasant


def test_validate_metric_input(simple_model_and_query):

    # only for testing, disable abstract methods.
    BaseMetric.__abstractmethods__ = set()

    base_metric = BaseMetric()

    base_metric.metric_template = (2, 3)
    base_metric.metric_name = "Example Metric"
    base_metric.metric_short_name = "EM"

    test_model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    query = Query(
        [flowers, insects],
        [pleasant, unpleasant],
        ["Flowers", "Weapons"],
        ["Pleasant", "Unpleasant"],
    )

    with pytest.raises(TypeError, match="query should be a Query instance, got*"):
        base_metric._check_input(None, test_model)

    with pytest.raises(
        TypeError, match="word_embedding should be a WordEmbeddingModel instance, got*"
    ):
        base_metric._check_input(query, None)

    query = Query(
        [flowers, insects, insects],
        [pleasant, unpleasant],
        ["Flowers", "Weapons", "Instruments"],
        ["Pleasant", "Unpleasant"],
    )
    with pytest.raises(
        Exception,
        match="The cardinality of the set of target words of the 'Flowers, Weapons and "
        "Instruments wrt Pleasant and Unpleasant' query does not match with the "
        "cardinality required by Example Metric. Provided query: 3, metric: 2",
    ):
        base_metric._check_input(query, test_model)

    query = Query(
        [flowers, insects],
        [pleasant, unpleasant],
        ["Flowers", "Weapons"],
        ["Pleasant", "Unpleasant"],
    )
    with pytest.raises(
        Exception,
        match="The cardinality of the set of attribute words of the 'Flowers and Weapons "
        "wrt Pleasant and Unpleasant' query does not match with the cardinality "
        "required by Example Metric. Provided query: 2, metric: 3",
    ):
        base_metric._check_input(query, test_model)


def test_run_query(simple_model_and_query):

    # only for testing, disable abstract methods.
    BaseMetric.__abstractmethods__ = set()

    base_metric = BaseMetric()

    base_metric.metric_template = (2, 2)
    base_metric.metric_name = "Example Metric"
    base_metric.metric_short_name = "EM"

    test_model, query, _, _, _, _ = simple_model_and_query

    with pytest.raises(NotImplementedError,):
        base_metric.run_query(query, test_model)
