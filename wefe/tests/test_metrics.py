import pytest
import numpy as np
from ..utils import load_weat_w2v
from ..word_embedding_model import WordEmbeddingModel
from ..datasets.datasets import load_weat
from ..query import Query
from ..metrics import WEAT, RND, RNSB, MAC, ECT
from sklearn.linear_model import LogisticRegression


def test_weat():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    weat = WEAT()
    query = Query([weat_word_set['flowers'], weat_word_set['insects']],
                  [weat_word_set['pleasant_5'], weat_word_set['unpleasant_5']],
                  ['Flowers', 'Insects'], ['Pleasant', 'Unpleasant'])
    results = weat.run_query(query, model)

    assert results[
        'query_name'] == 'Flowers and Insects wrt Pleasant and Unpleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))

    results = weat.run_query(query, model, return_effect_size=True)
    assert isinstance(results['result'], (np.float32, np.float64, float))


def test_rnd():

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    rnd = RND()
    query = Query([weat_word_set['flowers'], weat_word_set['insects']],
                  [weat_word_set['pleasant_5']], ['Flowers', 'Insects'],
                  ['Pleasant'])
    results = rnd.run_query(query, model)

    assert results['query_name'] == 'Flowers and Insects wrt Pleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))


def test_rnsb(capsys):

    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    rnsb = RNSB()
    query = Query([weat_word_set['flowers'], weat_word_set['insects']],
                  [weat_word_set['pleasant_5'], weat_word_set['unpleasant_5']],
                  ['Flowers', 'Insects'], ['Pleasant', 'Unpleasant'])
    results = rnsb.run_query(query, model)

    assert results[
        'query_name'] == 'Flowers and Insects wrt Pleasant and Unpleasant'
    assert list(results.keys()) == [
        'query_name', 'result', 'negative_sentiment_probabilities',
        'negative_sentiment_distribution'
    ]
    assert isinstance(results['result'], (np.float32, np.float64, float, np.float_))
    assert isinstance(results['negative_sentiment_probabilities'], dict)
    assert isinstance(results['negative_sentiment_distribution'], dict)

    query = Query([
        weat_word_set['flowers'], weat_word_set['instruments'],
        weat_word_set['male_terms'], weat_word_set['female_terms']
    ], [weat_word_set['pleasant_5'], weat_word_set['unpleasant_5']],
                  ['Flowers', 'Insects', 'Male terms', 'Female terms'],
                  ['Pleasant', 'Unpleasant'])
    results = rnsb.run_query(query, model)

    assert results[
        'query_name'] == 'Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))

    # custom classifier, print model eval and no params
    results = rnsb.run_query(query, model, classifier=LogisticRegression,
                             print_model_evaluation=True,
                             classifier_params=None)

    captured = capsys.readouterr()
    assert 'Classification Report' in captured.out

    assert results[
        'query_name'] == 'Flowers, Insects, Male terms and Female terms wrt Pleasant and Unpleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))

    # lost word threshold test
    results = rnsb.run_query(
        Query([['bla', 'asd'], weat_word_set['insects']],
              [weat_word_set['pleasant_5'], weat_word_set['unpleasant_5']],
              ['Flowers', 'Insects'], ['Pleasant', 'Unpleasant']), model)
    assert np.isnan(np.nan)


def test_mac():
    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    mac = MAC()
    query = Query(
        [weat_word_set['flowers']], [
            weat_word_set['pleasant_5'], weat_word_set['pleasant_9'],
            weat_word_set['unpleasant_5'], weat_word_set['unpleasant_9']
        ], ['Flowers'],
        ['Pleasant 5 ', 'Pleasant 9', 'Unpleasant 5', 'Unpleasant 9'])
    results = mac.run_query(query, model)

    assert results[
        'query_name'] == 'Flowers wrt Pleasant 5 , Pleasant 9, Unpleasant 5 and Unpleasant 9'
    assert isinstance(results['result'], (np.float32, np.float64, float))


def test_ect():
    weat_word_set = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

    ect = ECT()
    query = Query([weat_word_set['flowers'], weat_word_set['insects']],
                  [weat_word_set['pleasant_5']], ['Flowers', 'Insects'],
                  ['Pleasant'])
    results = ect.run_query(query, model)

    assert results['query_name'] == 'Flowers and Insects wrt Pleasant'
    assert isinstance(results['result'], (np.float32, np.float64, float))