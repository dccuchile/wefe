import pandas as pd
import logging
import numpy as np
import plotly.express as px

from gensim.models import KeyedVectors
from .word_embedding_model import WordEmbeddingModel
from .query import Query
from typing import Union
from wefe.metrics import MAC, RND, RNSB, WEAT, BaseMetric


def load_weat_w2v():
    from gensim.models import KeyedVectors
    # load dummy weat word vectors:
    weat_we = KeyedVectors.load_word2vec_format('./wefe/datasets/weat_w2v.txt', binary=False)
    return weat_we


## RUNNERS
def run_queries(metric: Union[MAC, RND, RNSB, WEAT, BaseMetric], queries: list, word_embeddings_models: list,
                queries_set_name: str = 'Unnamed queries set', lost_vocabulary_threshold: float = 0.2,
                metric_params: dict = {}, include_average_by_embedding: Union[None, str] = 'include',
                average_with_abs_values: bool = True) -> pd.DataFrame:
    """Run several queries over a several word embedding models using a specific metic.
    
    Parameters
    ----------
    metric : Union[MAC, RND, RNSB, WEAT, BaseMetric]
        A metric class.
    queries : list
        An iterable with a set of queries.
    word_embeddings_models : list
        An iterable with a set of word embedding pretrianed models.
    queries_set_name : str, optional
        The name of the set of queries or the criteria that will be tested, by default 'Unnamed queries set'
    lost_vocabulary_threshold : float, optional
        The threshold that will be passed to the , by default 0.2
    metric_params : dict, optional
        A dict with the given metric custom params if it needed, by default {}
    include_average_by_embedding : {None, 'include', 'only'}, optional
        It indicates if the result dataframe will include an average by model name of all calculated results, by default 'include'.
    average_with_abs_values : bool, optional
        Indicates if the average by embedding will be calculated from the absolute values of the results, by default True.

    Returns
    -------
    pd.DataFrame
        A dataframe with the results. The index contains the word embedding model name and the columns the experiment name. 
        Each cell represents the result of run a metric using a specific word embedding model and query.
    """

    # check inputs:

    # metric handling (TODO: issubclass not working...)
    # if not issubclass(metric, BaseMetric):
    # raise Exception('metric parameter must be instance of BaseMetric')

    # queries handling
    if not isinstance(queries, (list, np.ndarray)):
        raise TypeError('queries parameter must be a list or a numpy array. given: {}'.format(queries))
    if len(queries) == 0:
        raise Exception('queries list must have at least one query instance. given: {}'.format(queries))
    for idx, query in enumerate(queries):
        if query is None or not isinstance(query, Query):
            raise TypeError('item on index {} must be a Query instance. given: {}'.format(idx, query))

    # word vectors wrappers handling
    if not isinstance(word_embeddings_models, (list, np.ndarray)):
        raise TypeError(
            'word_vectors_wrappers parameter must be a list or a numpy array. given: {}'.format(word_embeddings_models))
    if len(word_embeddings_models) == 0:
        raise Exception('word_vectors_wrappers parameter must be a non empty list or numpy array. given: {}'.format(
            word_embeddings_models))
    for idx, word_vectors_wrapper in enumerate(word_embeddings_models):
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
    if not include_average_by_embedding in ['include', 'only'] and not include_average_by_embedding is None:
        raise Exception('return_averaged param must be any of {\'include\',\'only\', None}')

    # average_with_abs_values handling
    if not isinstance(average_with_abs_values, bool):
        raise Exception('average_with_abs_values param must be boolean')

    metric_instance = metric()
    results = []

    query_names = []

    for query in queries:
        for word_vector_wrapper in word_embeddings_models:
            result = metric_instance.run_query(query, word_vector_wrapper, lost_vocabulary_threshold, **metric_params)
            result['model_name'] = word_vector_wrapper.model_name
            results.append(result)

            if result['query_name'] not in query_names:
                query_names.append(result['query_name'])

    # get original column order
    # reorder the results in a legible table
    pivoted_results = pd.DataFrame(results).pivot(index='model_name', columns='query_name', values='result')
    pivoted_results = pivoted_results.reindex(index=[wvw.model_name for wvw in word_embeddings_models],
                                              columns=query_names)
    if include_average_by_embedding == 'include' or include_average_by_embedding == 'only':
        if average_with_abs_values:
            averaged_results = pd.DataFrame(pivoted_results.abs().mean(1))
        else:
            averaged_results = pd.DataFrame(pivoted_results.mean(1))
        averaged_results.columns = ['{}: {} average score'.format(metric_instance.metric_short_name, queries_set_name)]
        results = pd.concat([pivoted_results, averaged_results],
                            axis=1) if include_average_by_embedding == 'include' else averaged_results
        return results

    else:
        return pivoted_results


def plot_queries_results(results: pd.DataFrame, by: str = 'query'):
    """Plot the results obtained by a run_queries execution
    
    Parameters
    ----------
    results : pd.DataFrame
        A dataframe that contains the result of having executed run_queries with a set of queries and word embeddings.
    by : {'query', 'model'}, optional
        The aggregation function , by default 'query'
    
    Returns
    -------
    plotly.Figure
        A Figure that contains the generated graphic.
    
    Raises
    ------
    TypeError
        if results is not a instance of pandas DataFrame.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            'results must be a pandas DataFrame, result of having executed running_queries. Given: {}'.format(results))

    results_copy = results.copy(deep=True)

    if by == 'model':
        results_copy = results_copy
    else:
        results_copy = results_copy.T

    results_copy['query_name'] = results_copy.index

    cols = results_copy.columns
    id_vars = ['query_name']
    values_vars = [col_name for col_name in cols if col_name not in id_vars]

    # melt the dataframe
    melted_results = pd.melt(results_copy, id_vars=id_vars, value_vars=values_vars, var_name='Word Embedding Model')

    # configure the plot
    xaxis_title = 'Model' if by == 'model' else 'Query'

    fig = px.bar(melted_results, x='query_name', y="value", color='Word Embedding Model', barmode='group')
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Bias measure',
    )
    fig.for_each_trace(lambda t: t.update(name=" ".join(t.name.split('=')[1:])))  # delete Word Embedding Model = ...
    fig.for_each_trace(lambda t: t.update(x=['wrt<br>'.join(label.split('wrt')) for label in t.x]))
    # fig.show()
    return fig
