import pandas as pd
import logging
import numpy as np

from gensim.models import KeyedVectors
from .word_embedding_model import WordEmbeddingModel
from .query import Query


def load_weat_w2v():
    from gensim.models import KeyedVectors
    # load dummy weat word vectors:
    weat_we = KeyedVectors.load_word2vec_format('./wefe/datasets/weat_w2v.txt', binary=False)
    return weat_we


## RUNNERS


def run_queries(metric, queries, word_vectors_wrappers, queries_set_name='Unnamed queries set',
                lost_vocabulary_threshold=0.2, metric_params={}, return_averaged=None, average_with_abs_values=True):
    """
    Arguments:
        metric {[type]} -- Any type of metric that, when instantiated, can be called the run_experiment method.
        queries {list} -- A list with Experiment instances that will be runned.
        word_vectors_wrappers {WordVectorsWrappers} -- A list with WordVectorsWrappers instances that will be runned. 

    
    Keyword Arguments:
        experiment_name {str} -- [description] (default: {'Unnamed experiment})
        run_experiment_params {dict} -- [description] (default: {{}})
        return_averaged {str} -- {'include', 'only'} (default: {None})
        average_with_abs_values {bool} -- [description] (default: {True})
    """

    # check inputs:

    # metric handling
    if metric is None:
        raise Exception('metric parameter must not be None')

    # queries handling
    if not isinstance(queries, (list, np.ndarray)):
        raise TypeError('queries parameter must be a list or a numpy array. given: {}'.format(queries))
    if len(queries) == 0:
        raise Exception('queries list must have at least one query instance. given: {}'.format(queries))
    for idx, query in enumerate(queries):
        if query is None or not isinstance(query, Query):
            raise TypeError('item on index {} must be a Query instance. given: {}'.format(idx, query))

    # word vectors wrappers handling
    if not isinstance(word_vectors_wrappers, (list, np.ndarray)):
        raise TypeError(
            'word_vectors_wrappers parameter must be a list or a numpy array. given: {}'.format(word_vectors_wrappers))
    if len(word_vectors_wrappers) == 0:
        raise Exception('word_vectors_wrappers parameter must be a non empty list or numpy array. given: {}'.format(
            word_vectors_wrappers))
    for idx, word_vectors_wrapper in enumerate(word_vectors_wrappers):
        if word_vectors_wrapper is None or not isinstance(word_vectors_wrapper, WordEmbeddingModel):
            raise TypeError('item on index {} must be a WordVectorsWrapper instance. given: {}'.format(
                idx, word_vectors_wrapper))

    # experiment name handling
    if not isinstance(queries_set_name, str):
        raise Exception('queries_set_name parameter must be a non-empty string. given: {}'.format(queries_set_name))

    # metric_params handling
    if not isinstance(metric_params, dict):
        raise Exception('run_experiment_params must be a dict with a params for the metric')

    # return average handling
    if not return_averaged in ['include', 'only'] and not return_averaged is None:
        raise Exception('return_averaged param must be any of {\'include\',\'only\', None}')

    # average_with_abs_values handling
    if not isinstance(average_with_abs_values, bool):
        raise Exception('average_with_abs_values param must be boolean')

    metric_instance = metric()
    results = []

    exp_names = []

    for query in queries:
        for word_vector_wrapper in word_vectors_wrappers:
            result = metric_instance.run_query(query, word_vector_wrapper, lost_vocabulary_threshold, **metric_params)
            result['model_name'] = word_vector_wrapper.model_name
            results.append(result)

            if result['exp_name'] not in exp_names:
                exp_names.append(result['exp_name'])

    # get original column order
    # reorder the results in a legible table
    pivoted_results = pd.DataFrame(results).pivot(index='model_name', columns='exp_name', values='result')
    pivoted_results = pivoted_results.reindex(index=[wvw.model_name for wvw in word_vectors_wrappers],
                                              columns=exp_names)
    if return_averaged == 'include' or return_averaged == 'only':
        if average_with_abs_values:
            averaged_results = pd.DataFrame(pivoted_results.abs().mean(1))
        else:
            averaged_results = pd.DataFrame(pivoted_results.mean(1))
        averaged_results.columns = [
            '{}: {} average score'.format(metric_instance.abbreviated_method_name, queries_set_name)
        ]
        results = pd.concat([pivoted_results, averaged_results],
                            axis=1) if return_averaged == 'include' else averaged_results
        return results

    else:
        return pivoted_results


def graphic_results(results, ax=None, by='query'):
    import plotly.express as px

    if by == 'model':
        results = results
    else:
        results = results.T

    results['exp_name'] = results.index
    id_vars = ['exp_name']
    cols = results.columns
    values_vars = [col_name for col_name in cols if col_name not in id_vars]
    melted_results = pd.melt(results, id_vars=id_vars, value_vars=values_vars, var_name='Word Embedding Model')

    xaxis_title = 'Model' if by == 'model' else 'query'

    fig = px.bar(melted_results, x='exp_name', y="value", color='Word Embedding Model', barmode='group')
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Bias',
    )
    fig.for_each_trace(lambda t: t.update(name=t.name.split('=')[1]))
    fig.for_each_trace(lambda t: t.update(x=['wrt<br>'.join(label.split('wrt')) for label in t.x]))
    fig.show()
    return fig
