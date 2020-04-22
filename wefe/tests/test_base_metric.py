import pytest
from ..utils import load_weat_w2v
from ..metrics.base_metric import BaseMetric
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query


def test_create_base_metric():

    with pytest.raises(
            TypeError,
            match='Both components of template_required must be int or str*'):
        BaseMetric((None, 2), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of template_required must be int or str*'):
        BaseMetric((1, None), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of template_required must be int or str*'):
        BaseMetric(({}, 2), 'Example Metric', 'EM')

    with pytest.raises(
            TypeError,
            match='Both components of template_required must be int or str*'):
        BaseMetric((1, {}), 'Example Metric', 'EM')

    with pytest.raises(TypeError, match='metric_name must be a str*'):
        BaseMetric((1, 'n'), 1, 'EM')

    with pytest.raises(TypeError, match='metric_short_name must be a str*'):
        BaseMetric((1, 'n'), 'Example Metric', 0)

    base_metric = BaseMetric((2, 'n'), 'Example Metric', 'EM')
    assert base_metric.metric_template_ == (2, 'n')
    assert base_metric.metric_name_ == 'Example Metric'
    assert base_metric.metric_short_name_ == 'EM'


def test_get_embeddings_from_word_set():

    base_metric = BaseMetric((2, 2), 'Example Metric', 'EM')
    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    flowers = weat['flowers']
    weapons = weat['weapons']
    pleasant = weat['pleasant_5']
    unpleasant = weat['unpleasant_5']
    query = Query([flowers, weapons], [pleasant, unpleasant],
                  ['Flowers', 'Weapons'], ['Pleasant', 'Unpleasant'])

    embeddings = base_metric._get_embeddings_from_query(query, model)
    target_embeddings, attribute_embeddings = embeddings

    assert len(target_embeddings) == 2
    assert len(attribute_embeddings) == 2

    assert len(target_embeddings[0].keys()) == len(flowers)
    assert len(target_embeddings[1].keys()
               ) == len(weapons) - 1  # word axe is not in the set
    assert len(attribute_embeddings[0].keys()) == len(pleasant)
    assert len(attribute_embeddings[1].keys()) == len(unpleasant)


def test_validate_metric_input():
    base_metric = BaseMetric((2, 3), 'Example Metric', 'EM')
    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    flowers = weat['flowers']
    weapons = weat['weapons']
    instruments = weat['instruments']
    pleasant = weat['pleasant_5']
    unpleasant = weat['unpleasant_5']

    query = Query([flowers, weapons], [pleasant, unpleasant],
                  ['Flowers', 'Weapons'], ['Pleasant', 'Unpleasant'])

    with pytest.raises(TypeError,
                       match='query parameter must be a Query instance.'):
        base_metric._check_input(None, model, 0.2, True)

    with pytest.raises(
            TypeError,
            match='word_embedding must be a WordEmbeddingModel instance'):
        base_metric._check_input(query, None, 0.2, True)

    with pytest.raises(TypeError,
                       match='lost_vocabulary_threshold must be a float.'):
        base_metric._check_input(query, model, '0.2', True)

    with pytest.raises(TypeError,
                       match='warn_filtered_words must be a bool. '):
        base_metric._check_input(query, model, 0.2, None)

    query = Query([flowers, weapons, instruments], [pleasant, unpleasant],
                  ['Flowers', 'Weapons', 'Instruments'],
                  ['Pleasant', 'Unpleasant'])
    with pytest.raises(
            Exception,
            match='The cardinality of the set of target words of the'
            ' provided query*'):
        base_metric._check_input(query, model, 0.2, False)

    query = Query([flowers, weapons], [pleasant, unpleasant],
                  ['Flowers', 'Weapons'], ['Pleasant', 'Unpleasant'])
    with pytest.raises(Exception,
                       match='The cardinality of the set of attributes'
                       ' words of the provided query*'):
        base_metric._check_input(query, model, 0.2, False)


def test_some_set_has_fewer_words_than_the_threshold():
    base_metric = BaseMetric((2, 2), 'Example Metric', 'EM')
    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    flowers = weat['flowers']
    weapons = weat['weapons']
    pleasant = weat['pleasant_5']
    unpleasant = weat['unpleasant_5']

    query = Query([flowers, ['bla', 'asd']], [pleasant, unpleasant],
                  ['Flowers', 'bla'], ['Pleasant', 'Unpleasant'])
    assert base_metric._get_embeddings_from_query(query, model) is None
    query = Query([['bla', 'asd'], weapons], [pleasant, unpleasant],
                  ['Flowers', 'bla'], ['Pleasant', 'Unpleasant'])
    assert base_metric._get_embeddings_from_query(query, model) is None
    query = Query([flowers, weapons], [['bla', 'asd'], unpleasant],
                  ['Flowers', 'Weapons'], ['bla', 'Unpleasant'])
    assert base_metric._get_embeddings_from_query(query, model) is None
    query = Query([flowers, weapons], [pleasant, ['bla', 'asd']],
                  ['Flowers', 'Weapons'], ['Pleasant', 'bla'])
    assert base_metric._get_embeddings_from_query(query, model) is None
