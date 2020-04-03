import pandas as pd
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from gensim.models import KeyedVectors

from .word_embedding_model import WordEmbeddingModel
from .query import Query
from typing import Union, Iterable
from wefe.metrics import MAC, RND, RNSB, WEAT

# -----------------------------------------------------------------------------
# ---------------------------------- Runners ----------------------------------
# -----------------------------------------------------------------------------


def run_queries(metric: Union[MAC, RND, RNSB, WEAT], queries: list,
                word_embeddings_models: list,
                queries_set_name: str = 'Unnamed queries set',
                lost_vocabulary_threshold: float = 0.2,
                metric_params: dict = {},
                include_average_by_embedding: Union[None, str] = 'include',
                average_with_abs_values: bool = True,
                warn_filtered_words: bool = False) -> pd.DataFrame:
    """Run several queries over a several word embedding models using a specific metic.
    
    Parameters
    ----------
    metric : Union[MAC, RND, RNSB, WEAT]
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
        raise TypeError(
            'queries parameter must be a list or a numpy array. given: {}'.
            format(queries))
    if len(queries) == 0:
        raise Exception(
            'queries list must have at least one query instance. given: {}'.
            format(queries))

    for idx, query in enumerate(queries):
        if query is None or not isinstance(query, Query):
            raise TypeError(
                'item on index {} must be a Query instance. given: {}'.format(
                    idx, query))

    # word vectors wrappers handling
    if not isinstance(word_embeddings_models, (list, np.ndarray)):
        raise TypeError(
            'word_embeddings_models parameter must be a list or a numpy array. given: {}'
            .format(word_embeddings_models))

    if len(word_embeddings_models) == 0:
        raise Exception(
            'word_embeddings_models parameter must be a non empty list or numpy array. given: {}'
            .format(word_embeddings_models))

    for idx, model in enumerate(word_embeddings_models):
        if model is None or not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                'item on index {} must be a WordEmbeddingModel instance. given: {}'
                .format(idx, model))

    # experiment name handling
    if not isinstance(queries_set_name, str) or queries_set_name == '':
        raise TypeError(
            'When queries_set_name parameter is provided, it must be a non-empty string. given: {}'
            .format(queries_set_name))

    # metric_params handling
    if not isinstance(metric_params, dict):
        raise TypeError(
            'run_experiment_params must be a dict with a params for the metric'
        )

    # return average handling
    if not include_average_by_embedding in ['include', 'only', None]:
        raise Exception(
            "include_average_by_embedding param must be any of 'include','only', None. Given: {}"
            .format(include_average_by_embedding))

    # average_with_abs_values handling
    if not isinstance(average_with_abs_values, bool):
        raise TypeError(
            'average_with_abs_values param must be boolean. Given: {}'.format(
                average_with_abs_values))

    metric_instance = metric()
    results = []

    query_names = []

    for query in queries:
        for model in word_embeddings_models:
            result = metric_instance.run_query(
                query, model,
                lost_vocabulary_threshold=lost_vocabulary_threshold,
                warn_filtered_words=warn_filtered_words, **metric_params)
            result['model_name'] = model.model_name_
            results.append(result)

            if result['query_name'] not in query_names:
                query_names.append(result['query_name'])

    # get original column order
    # reorder the results in a legible table
    pivoted_results = pd.DataFrame(results).pivot(index='model_name',
                                                  columns='query_name',
                                                  values='result')
    pivoted_results = pivoted_results.reindex(
        index=[model.model_name_ for model in word_embeddings_models],
        columns=query_names)
    if include_average_by_embedding == 'include' or include_average_by_embedding == 'only':
        if average_with_abs_values:
            averaged_results = pd.DataFrame(pivoted_results.abs().mean(1))
        else:
            averaged_results = pd.DataFrame(pivoted_results.mean(1))
        averaged_results.columns = [
            '{}: {} average score'.format(metric_instance.metric_short_name_,
                                          queries_set_name)
        ]
        results = pd.concat(
            [pivoted_results, averaged_results], axis=1
        ) if include_average_by_embedding == 'include' else averaged_results
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
            'results must be a pandas DataFrame, result of having executed running_queries. Given: {}'
            .format(results))

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
    melted_results = pd.melt(results_copy, id_vars=id_vars,
                             value_vars=values_vars,
                             var_name='Word Embedding Model')

    # configure the plot
    xaxis_title = 'Model' if by == 'model' else 'Query'

    fig = px.bar(melted_results, x='query_name', y="value",
                 color='Word Embedding Model', barmode='group')
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Bias measure',
    )
    fig.for_each_trace(lambda t: t.update(name=" ".join(t.name.split('=')[1:]))
                       )  # delete Word Embedding Model = ...
    fig.for_each_trace(lambda t: t.update(
        x=['wrt<br>'.join(label.split('wrt')) for label in t.x]))
    # fig.show()
    return fig


# -----------------------------------------------------------------------------
# --------------------------------- Rankings ----------------------------------
# -----------------------------------------------------------------------------


def create_ranking(results_dataframes: Iterable[pd.DataFrame]):
    """Creates a ranking form the average scores of the provided dataframes.
    
    Parameters
    ----------
    results_dataframes : Iterable[pd.DataFrame]
        A list or array of dataframes returned by the run_queries function.
    
    Returns
    -------
    pd.DataFrame
        A dataframe with the ranked average scores.
    
    Raises
    ------
    Exception
        If there is no average column in some result Dataframe.
    TypeError
        If some element of results_dataframes is not a pandas DataFrame.
    """
    def get_average_scores(results_dataframes: list):
        remaining_columns = []

        for idx, results in enumerate(results_dataframes):
            is_average_inside_result_df = False

            # find the col with that contains the average word inside.
            for col in results.columns:
                if 'average' in col:
                    remaining_columns.append(results[[col]])
                    is_average_inside_result_df = True

            # if the result dataframe does not have a average column, raise a exception.
            if is_average_inside_result_df == False:
                raise Exception(
                    'There is no average column in the {} dataframe\n{}'.
                    format(idx, results))

        return pd.concat(remaining_columns, axis=1)

    # check the input.
    for idx, results_df in enumerate(results_dataframes):
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(
                'All elements of results_dataframes must be a pandas Dataframe instance. Given at position {}: {}'
                .format(idx, results_df))
    # get the avg_scores columns and merge into one dataframe
    avg_scores = get_average_scores(results_dataframes)

    rankings = []
    for col in avg_scores:
        # for each avg_score column, calculate the ranking
        rankings.append(avg_scores[col].values.argsort(axis=0).argsort(
            axis=0) + 1)
    return pd.DataFrame(rankings, columns=avg_scores.index,
                        index=avg_scores.columns).T


def plot_ranking(ranking: pd.DataFrame, title: str = '',
                 use_metric_as_facet: bool = True):
    def melt_df(results):
        results = results.copy()
        results['exp_name'] = results.index
        id_vars = ['exp_name']
        cols = results.columns
        values_vars = [
            col_name for col_name in cols if col_name not in id_vars
        ]
        melted_results = pd.melt(results, id_vars=id_vars,
                                 value_vars=values_vars, var_name='Metric')
        melted_results.columns = ['Embedding model', 'Metric', 'Ranking']
        return melted_results

    melted_ranking = melt_df(ranking.copy(deep=True))

    if use_metric_as_facet == True:
        fig = px.bar(melted_ranking, x="Ranking", y="Embedding model",
                     barmode="stack", color='Metric', orientation='h',
                     facet_col='Metric')
    else:
        fig = px.bar(melted_ranking, x="Ranking", y="Embedding model",
                     barmode="stack", color='Metric', orientation='h')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text='')
    fig.update_yaxes(tickfont=dict(size=10))
    fig.for_each_trace(lambda t: t.update(name=t.name.split('=')[1]))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    return fig


# -----------------------------------------------------------------------------
# ------------------------------- Correlations --------------------------------
# -----------------------------------------------------------------------------


def calculate_ranking_correlations(ranking_dataframe,
                                   correlation_function=stats.spearmanr):

    # correlation_function = 'stats.kendaltau'

    matrix = []

    for idx_1, _ in enumerate(ranking_dataframe.columns):
        matrix.append([])
        for idx_2, _ in enumerate(ranking_dataframe.columns):
            matrix[idx_1].append(
                correlation_function(
                    ranking_dataframe.iloc[:, [idx_1]],
                    ranking_dataframe.iloc[:, [idx_2]]).correlation)

    correlation_matrix = pd.DataFrame(matrix)
    correlation_matrix.columns = ranking_dataframe.columns
    correlation_matrix.index = ranking_dataframe.columns
    return correlation_matrix


def plot_ranking_correlations(correlation_matrix, title=''):
    fig = go.Figure(
        data=go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns,
                        y=correlation_matrix.index, hoverongaps=False,
                        zmin=0.0, zmax=1, colorscale='Darkmint'))
    fig.update_layout(title=title, font=dict(color="#000000"))
    return fig


def load_weat_w2v():
    from gensim.models import KeyedVectors
    # load dummy weat word vectors:
    weat_we = KeyedVectors.load_word2vec_format('./wefe/datasets/weat_w2v.txt',
                                                binary=False)
    return weat_we