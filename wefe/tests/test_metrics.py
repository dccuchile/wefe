import pytest
import numpy as np
from ..utils import load_weat_w2v
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query
from ..metrics.WEAT import WEAT
from ..metrics.RND import RND
from ..metrics.RNSB import RNSB


def test_weat():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    weat = WEAT()
    query = Query([weat_word_set['Flowers'], weat_word_set['Insects']],
                  [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    results = weat.run_query(query, model)

    assert results['query_name'] == 'Flowers and Insects wrt Pleasant and Unpleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))

    results = weat.run_query(query, model, return_effect_size=True)
    assert isinstance(results['result'], (np.float32, np.float64, float))


def test_rnd():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    rnd = RND()
    query = Query([weat_word_set['Flowers'], weat_word_set['Insects']], [weat_word_set['Pleasant 5']],
                  ['Flowers', 'Insects'], ['Pleasant'])
    results = rnd.run_query(query, model)

    assert results['query_name'] == 'Flowers and Insects wrt Pleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))


def test_rnsb():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    rnsb = RNSB()
    query = Query([weat_word_set['Flowers'], weat_word_set['Insects']],
                  [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
                  ['Pleasant', 'Unpleasant'])
    results = rnsb.run_query(query, model)

    assert results['query_name'] == 'Flowers and Insects wrt Pleasant and Unpleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))
