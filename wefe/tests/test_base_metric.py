import pytest
from ..utils import load_weat_w2v
from ..metrics.base_metric import BaseMetric
from ..word_embedding import WordEmbedding
from ..datasets.datasets import load_weat
from ..query import Query


@pytest.fixture
def simple_model_and_query():
    w2v = load_weat_w2v()
    model = WordEmbedding(w2v, 'weat_w2v', '')
    weat_wordsets = load_weat()

    flowers = weat_wordsets['flowers']
    insects = weat_wordsets['insects']
    pleasant = weat_wordsets['pleasant_5']
    unpleasant = weat_wordsets['unpleasant_5']
    query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    return w2v, model, query, flowers, insects, pleasant, unpleasant


def test_create_base_metric():

    # only for testing, disable abstract methods.
    BaseMetric.__abstractmethods__ = set()

    with pytest.raises(
            TypeError,
            match='Both components of metric_template should be a int or str*'):
        BaseMetric((None, 2), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of metric_template should be a int or str*'):
        BaseMetric((1, None), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of metric_template should be a int or str*'):
        BaseMetric(({}, 2), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of metric_template should be a int or str*'):
        BaseMetric((1, {}), 'Example Metric', 'EM')

    with pytest.raises(TypeError, match='metric_name should be a str*'):
        BaseMetric((1, 'n'), 1, 'EM')

    with pytest.raises(TypeError, match='metric_short_name should be a str*'):
        BaseMetric((1, 'n'), 'Example Metric', 0)

    base_metric = BaseMetric((2, 'n'), 'Example Metric', 'EM')
    assert base_metric.metric_template == (2, 'n')
    assert base_metric.metric_name == 'Example Metric'
    assert base_metric.metric_short_name == 'EM'


def test_validate_metric_input(simple_model_and_query):

    # only for testing, disable abstract methods.
    BaseMetric.__abstractmethods__ = set()

    base_metric = BaseMetric((2, 3), 'Example Metric', 'EM')
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Weapons'],
                  ['Pleasant', 'Unpleasant'])

    with pytest.raises(TypeError, match='query should be a Query instance, got*'):
        base_metric._check_input(None, model)

    with pytest.raises(TypeError,
                       match='word_embedding should be a WordEmbedding instance, got*'):
        base_metric._check_input(query, None)

    query = Query([flowers, insects, insects], [pleasant, unpleasant],
                  ['Flowers', 'Weapons', 'Instruments'], ['Pleasant', 'Unpleasant'])
    with pytest.raises(
            Exception,
            match=
            "The cardinality of the set of target words of the 'Flowers, Weapons and "
            "Instruments wrt Pleasant and Unpleasant' query does not match with the "
            "cardinality required by Example Metric. Provided query: 3, metric: 2"):
        base_metric._check_input(query, model)

    query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Weapons'],
                  ['Pleasant', 'Unpleasant'])
    with pytest.raises(
            Exception,
            match=
            "The cardinality of the set of attribute words of the 'Flowers and Weapons "
            "wrt Pleasant and Unpleasant' query does not match with the cardinality "
            "required by Example Metric. Provided query: 2, metric: 3"):
        base_metric._check_input(query, model)


def test_run_query(simple_model_and_query):

    # only for testing, disable abstract methods.
    BaseMetric.__abstractmethods__ = set()

    base_metric = BaseMetric((2, 2), 'Example Metric', 'EM')
    w2v, model, query, flowers, insects, pleasant, unpleasant = simple_model_and_query

    assert base_metric.run_query(query, model) == {}
