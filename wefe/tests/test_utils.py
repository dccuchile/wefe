import pytest
from ..utils import load_weat_w2v, get_embeddings_from_word_set, verify_metric_input
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query


def test_get_embeddings_from_word_set():

    weat = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')
    # load embeddings. axe doesn't exists in the vocab. it is filtered.
    embeddings, filtered_words = get_embeddings_from_word_set(weat['Weapons'], model, True)

    assert filtered_words == ['axe']
    assert len(embeddings) == len(weat['Weapons']) - 1

    # type errors
    with pytest.raises(TypeError, match='word_set must be a list or a numpy array.'):
        embeddings, filtered_words = get_embeddings_from_word_set(None, model)

    with pytest.raises(TypeError, match='word_embedding must be a instance of WordEmbeddingModel class'):
        embeddings, filtered_words = get_embeddings_from_word_set(weat['Weapons'], None)

    with pytest.raises(TypeError, match='warn_filtered_words must be a boolean'):
        embeddings, filtered_words = get_embeddings_from_word_set(weat['Weapons'], model, None)


def test_validate_metric_input():
    weat_word_set = load_weat()
    w2v = load_weat_w2v()
    model = WordEmbeddingModel(w2v, 'weat_w2v', '')

    # load embeddings. axe doesn't exists in the vocab. it is filtered.
    query = Query([weat_word_set['Flowers'], weat_word_set['Insects']],
                  [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])

    with pytest.raises(TypeError, match='query parameter must be a Query instance.'):
        verify_metric_input(None, model, (2, 2), 'WEAT')

    with pytest.raises(TypeError, match='word_embedding must be a WordVectorsWrapper instance'):
        verify_metric_input(query, None, (2, 2), 'WEAT')

    with pytest.raises(TypeError, match='Both components of template_required must be int or str.'):
        verify_metric_input(query, model, (None, 2), 'WEAT')

    with pytest.raises(TypeError, match='Both components of template_required must be int or str.'):
        verify_metric_input(query, model, (2, None), 'WEAT')

    with pytest.raises(Exception, match='The cardinality of the set of target words of the provided query'):
        verify_metric_input(query, model, (3, 2), 'WEAT')

    with pytest.raises(Exception, match='The cardinality of the set of attributes words of the provided query'):
        verify_metric_input(query, model, (2, 3), 'WEAT')
