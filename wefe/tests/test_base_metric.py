import pytest
from ..utils import load_weat_w2v
from ..metrics.metric import BaseMetric
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query


def test_get_embeddings_from_word_set():

    base_metric = BaseMetric((2, 3), 'Example Metric', 'EM')
    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    flowers = weat['Flowers']
    weapons = weat['Weapons']
    pleasant = weat['Pleasant 5']
    unpleasant = weat['Unpleasant 5']
    query = Query([flowers, weapons], [pleasant, unpleasant], ['Flowers', 'Weapons'], ['Pleasant', 'Unpleasant'])

    target_embeddings, attribute_embeddings = base_metric._get_embeddings_from_query(query, model)

    assert len(target_embeddings) == 2
    assert len(attribute_embeddings) == 2

    assert len(target_embeddings[0].keys()) == len(flowers)
    assert len(target_embeddings[1].keys()) == len(weapons) - 1  # word axe is not in the set
    assert len(attribute_embeddings[0].keys()) == len(pleasant)
    assert len(attribute_embeddings[1].keys()) == len(unpleasant)


def test_validate_metric_input():
    base_metric = BaseMetric((2, 3), 'Example Metric', 'EM')
    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    flowers = weat['Flowers']
    weapons = weat['Weapons']
    pleasant = weat['Pleasant 5']
    unpleasant = weat['Unpleasant 5']
    query = Query([flowers, weapons], [pleasant, unpleasant], ['Flowers', 'Weapons'], ['Pleasant', 'Unpleasant'])

    with pytest.raises(TypeError, match='query parameter must be a Query instance.'):
        base_metric._check_input(None, model, 0.2, True)

    with pytest.raises(TypeError, match='word_embedding must be a WordEmbeddingModel instance'):
        base_metric._check_input(query, None, 0.2, True)

    with pytest.raises(TypeError, match='lost_vocabulary_threshold must be a float.'):
        base_metric._check_input(query, model, '0.2', True)

    with pytest.raises(TypeError, match='warn_filtered_words must be a bool. '):
        base_metric._check_input(query, model, 0.2, None)
