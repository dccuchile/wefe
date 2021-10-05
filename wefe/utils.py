"""
A collection of WEFE utility functions.

This file contains functions for to process to massively execute queries, aggregate them
through rankings and graph these results.
"""

import copy
import logging
from typing import Callable, Dict, List, Type, Union

import numpy as np
import pandas as pd
import pkg_resources
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def check_is_fitted(estimator, attributes):
    msg = (
        "This %(name)s instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this debias method."
    )
    _check_is_fitted(estimator, attributes, msg=msg)


# -----------------------------------------------------------------------------
# ---------------------------------- Runners ----------------------------------
# -----------------------------------------------------------------------------


AGGREGATION_FUNCTIONS = {
    "sum": lambda df: df.sum(1),
    "avg": lambda df: df.mean(1),
    "abs_sum": lambda df: df.abs().sum(1),
    "abs_avg": lambda df: df.abs().mean(1),
}

AGGREGATION_FUNCTION_NAMES = {
    "sum": "sum",
    "avg": "average",
    "abs_sum": "sum of abs values",
    "abs_avg": "average of abs values",
}


def generate_subqueries_from_queries_list(
    metric: Type[BaseMetric], queries: List[Query]
) -> List[Query]:
    """Generate a list of subqueries from queries.

    Parameters
    ----------
    metric : Type[BaseMetric]
        Some metric.
    queries : List[Query]
        A list with queries.

    Returns
    -------
    List[Query]
        A list with all the generated subqueries.
    """
    # instance metric
    metric = metric()

    subqueries = []
    for query_idx, query in enumerate(queries):
        try:
            subqueries += query.get_subqueries(metric.metric_template)
        except Exception as e:
            logging.warning(
                "Query in index {} ({}) can not be splitted in subqueries "
                "with the {} metric template = {}. Exception: \n{}".format(
                    query_idx,
                    query.query_name,
                    metric.metric_name,
                    metric.metric_template,
                    e,
                )
            )

    # remove duplicates (o(n^2)...)
    filtered_subqueries: List[Query] = []
    for subquery in subqueries:
        duplicated = False
        for filtered_subquery in filtered_subqueries:
            if filtered_subquery.query_name == subquery.query_name:
                duplicated = True
                break
        if not duplicated:
            filtered_subqueries.append(subquery)

    return filtered_subqueries


def run_queries(
    metric: Type[BaseMetric],
    queries: List[Query],
    models: List[WordEmbeddingModel],
    queries_set_name: str = "Unnamed queries set",
    lost_vocabulary_threshold: float = 0.2,
    metric_params: dict = {},
    generate_subqueries: bool = False,
    aggregate_results: bool = False,
    aggregation_function: Union[str, Callable] = "abs_avg",
    return_only_aggregation: bool = False,
    warn_not_found_words: bool = False,
) -> pd.DataFrame:
    """Run several queries over a several word embedding models using a specific metic.

    Parameters
    ----------
    metric : Type[BaseMetric]
        A metric class.
    queries : list
        An iterable with a set of queries.
    word_embeddings_models : list
        An iterable with a set of word embedding pretrianed models.
    queries_set_name : str, optional
        The name of the set of queries or the criteria that will be tested,
        by default 'Unnamed queries set'
    lost_vocabulary_threshold : float, optional
        The threshold that will be passed to the , by default 0.2
    metric_params : dict, optional
        A dict with custom params that will passed to run_query method of the
        respective metric, by default {}
    generate_subqueries: bool, optional
        It indicates if the program, when detecting queries with a bigger
        template than the metric, should try to generate subqueries compatible
        with it.
        If any query is compatible with the metric template, then it appends
        the same query.
        DANGER: This may cause some comparisons to become meaningless when
        comparing biases that are not compatible with each other.
        By default, False.
    aggregate_results : bool, optional
        A boolean that indicates if the results must be aggregated with some
        function.
    aggregation_function : Union[str, Callable], optional
        The function that will be applied row by row to add the results.
        It must be pandas row compatible operation.
        Implemented functions: 'sum', 'abs_sub', 'avg' and 'abs_avg',
        by default 'abs_avg'.
    return_only_aggregation : bool, optional
        If return_only_aggregation is True, only the column with the added queries is
        returned, by default False.

    Returns
    -------
    pd.DataFrame
        A dataframe with the results. The index contains the word embedding model name
        and the columns the experiment name.
        Each cell represents the result of run a metric using a specific word
        embedding model and query.
    """
    # check inputs:

    # metric handling (TODO: issubclass not working...)
    # if not issubclass(metric, BaseMetric):
    # raise Exception('metric parameter must be instance of BaseMetric')

    # queries handling
    if not isinstance(queries, (list, np.ndarray)):
        raise TypeError(
            "queries parameter must be a list or a numpy array. given: {}".format(
                queries
            )
        )
    if len(queries) == 0:
        raise Exception(
            "queries list must have at least one query instance. given: {}".format(
                queries
            )
        )

    for idx, query in enumerate(queries):
        if query is None or not isinstance(query, Query):
            raise TypeError(
                "item on index {} must be a Query instance. given: {}".format(
                    idx, query
                )
            )

    # word vectors wrappers handling
    if not isinstance(models, (list, np.ndarray)):
        raise TypeError(
            "word_embeddings_models parameter must be a list or a numpy array."
            " given: {}".format(models)
        )

    if len(models) == 0:
        raise Exception(
            "word_embeddings_models parameter must be a non empty list or "
            "numpy array. given: {}".format(models)
        )

    for idx, model in enumerate(models):
        if model is None or not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                "item on index {} must be a WordEmbeddingModel instance. "
                "given: {}".format(idx, model)
            )

    # experiment name handling
    if not isinstance(queries_set_name, str) or queries_set_name == "":
        raise TypeError(
            "When queries_set_name parameter is provided, it must be a "
            "non-empty string. given: {}".format(queries_set_name)
        )

    # metric_params handling
    if not isinstance(metric_params, dict):
        raise TypeError(
            "run_experiment_params must be a dict with a params for the metric"
        )

    # aggregate results bool
    if not isinstance(aggregate_results, bool):
        raise TypeError(
            "aggregate_results parameter must be a bool value. Given:"
            "{}".format(aggregate_results)
        )

    # aggregation function:
    AGG_FUNCTION_MSG = (
        "aggregation_function must be one of 'sum',"
        "abs_sum', 'avg', 'abs_avg' or a callable. given: {}"
    )
    if isinstance(aggregation_function, str):
        if aggregation_function not in ["sum", "abs_sum", "avg", "abs_avg"]:
            raise Exception(AGG_FUNCTION_MSG.format(aggregation_function))
    elif not callable(aggregation_function):
        raise Exception(AGG_FUNCTION_MSG.format(aggregation_function))

    # average_with_abs_values handling
    if not isinstance(return_only_aggregation, bool):
        raise TypeError(
            "return_only_aggregation param must be boolean. Given: {}".format(
                return_only_aggregation
            )
        )

    if generate_subqueries:
        queries = generate_subqueries_from_queries_list(metric, queries)

    metric_instance = metric()
    results = []

    query_names = []
    try:
        for query in queries:
            for model in models:
                result = metric_instance.run_query(
                    query,
                    model,
                    lost_vocabulary_threshold=lost_vocabulary_threshold,
                    warn_not_found_words=warn_not_found_words,
                    **metric_params,
                )
                result["model_name"] = model.name
                results.append(result)

                if result["query_name"] not in query_names:
                    query_names.append(result["query_name"])

    except Exception as e:
        raise Exception(
            "Error during executing the query {} on the model {}: {}".format(
                query.query_name, model.name, str(e)
            )
        )

    # get original column order
    # reorder the results in a legible table
    pivoted_results = pd.DataFrame(results).pivot(
        index="model_name", columns="query_name", values="result"
    )
    pivoted_results = pivoted_results.reindex(
        index=[model.name for model in models], columns=query_names,
    )

    if aggregate_results:

        # if the aggregation function is one of the preimplemented functions.
        if (
            isinstance(aggregation_function, str)
            and aggregation_function in AGGREGATION_FUNCTIONS
        ):
            aggregated_results = AGGREGATION_FUNCTIONS[aggregation_function](
                pivoted_results
            )
            aggregated_results_name = AGGREGATION_FUNCTION_NAMES[aggregation_function]

        # run the custom aggregation function over the pivoted results
        else:
            aggregated_results = aggregation_function(pivoted_results)
            aggregated_results_name = "custom aggregation"

        # generate the new aggregation column name.
        aggregation_column_name = "{}: {} {} score".format(
            metric_instance.metric_short_name, queries_set_name, aggregated_results_name
        )

        # set the aggregation column name.
        aggregated_results = pd.DataFrame(
            aggregated_results, columns=[aggregation_column_name]
        )

        # return option with only aggregation.
        if return_only_aggregation:
            return aggregated_results

        results = pd.concat([pivoted_results, aggregated_results], axis=1)
        return results

    return pivoted_results


# -----------------------------------------------------------------------------
# ----------------------------------- Plots -----------------------------------
# -----------------------------------------------------------------------------


def plot_queries_results(results: pd.DataFrame, by: str = "query"):
    """Plot the results obtained by a run_queries execution.

    Parameters
    ----------
    results : pd.DataFrame
        A dataframe that contains the result of having executed run_queries
        with a set of queries and word embeddings.
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
            "results must be a pandas DataFrame, result of having executed "
            "running_queries. Given: {}".format(results)
        )

    results_copy = results.copy(deep=True)

    if by == "model":
        results_copy = results_copy
    else:
        results_copy = results_copy.T

    results_copy["query_name"] = results_copy.index

    cols = results_copy.columns
    id_vars = ["query_name"]
    values_vars = [col_name for col_name in cols if col_name not in id_vars]

    # melt the dataframe
    melted_results = pd.melt(
        results_copy,
        id_vars=id_vars,
        value_vars=values_vars,
        var_name="Word Embedding Model",
    )

    # configure the plot
    xaxis_title = "Model" if by == "model" else "Query"

    fig = px.bar(
        melted_results,
        x="query_name",
        y="value",
        color="Word Embedding Model",
        barmode="group",
    )
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title="Bias measure",
    )
    fig.for_each_trace(
        lambda t: t.update(x=["wrt<br>".join(label.split("wrt")) for label in t.x])
    )
    # fig.show()
    return fig


# -----------------------------------------------------------------------------
# --------------------------------- Rankings ----------------------------------
# -----------------------------------------------------------------------------


def create_ranking(
    results_dataframes: List[pd.DataFrame],
    method: str = "first",
    ascending: bool = True,
):
    """Create a ranking form the aggregated scores of the provided dataframes.

    The function will assume that the aggregated scores are in the last column
    of each result dataframe.
    It uses pandas.DataFrame.rank to generate the ranks.

    Parameters
    ----------
    results_dataframes : List[pd.DataFrame]
        A list or array of dataframes returned by the run_queries function.

    method : str, optional
        How to rank the group of records that have the same value, by default 'first'.
        The options are:
        - average: average rank of the group
        - min: lowest rank in the group
        - max: highest rank in the group
        - first: ranks assigned in order they appear in the array
        - dense: like ‘min’, but rank always increases by 1 between groups.

    ascending : bool, optional
        Whether or not the elements should be ranked in ascending order, by default True.

    Returns
    -------
    pd.DataFrame
        A dataframe with the ranked scores.

    Raises
    ------
    Exception
        If there is no average column in some result Dataframe.
    TypeError
        If some element of results_dataframes is not a pandas DataFrame.
    """
    # check the input.
    for idx, results_df in enumerate(results_dataframes):
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(
                "All elements of results_dataframes must be a pandas "
                "Dataframe instance. Got {} at position {}".format(
                    type(results_df), idx
                )
            )

    # get the avg_scores columns and merge into one dataframe
    aggregation_columns: List[pd.DataFrame] = []

    for result in results_dataframes:
        aggregation_columns.append(result[result.columns[-1]])

    # check for duplicated column names
    column_names = pd.Series([series.name for series in aggregation_columns])
    duplicated_names = column_names[column_names.duplicated(keep="first")]

    no_duplicated_column_names = copy.copy(column_names)
    for duplicated_name in duplicated_names:
        count = 0
        for idx, name in enumerate(no_duplicated_column_names):
            if name == duplicated_name:
                no_duplicated_column_names[idx] = "{} ({})".format(name, idx + 1)
                count += 1

    avg_scores = pd.concat(aggregation_columns, axis=1)
    avg_scores.columns = no_duplicated_column_names
    rankings = avg_scores.rank(method=method, ascending=ascending)
    return rankings


def plot_ranking(
    ranking: pd.DataFrame, title: str = "", use_metric_as_facet: bool = False
):
    def melt_df(results):
        results = results.copy()
        results["exp_name"] = results.index
        id_vars = ["exp_name"]
        cols = results.columns
        values_vars = [col_name for col_name in cols if col_name not in id_vars]
        melted_results = pd.melt(
            results, id_vars=id_vars, value_vars=values_vars, var_name="Metric"
        )
        melted_results.columns = ["Embedding model", "Metric", "Ranking"]
        return melted_results

    melted_ranking = melt_df(ranking.copy(deep=True))

    if use_metric_as_facet:
        fig = px.bar(
            melted_ranking,
            x="Ranking",
            y="Embedding model",
            barmode="stack",
            color="Metric",
            orientation="h",
            facet_col="Metric",
        )
    else:
        fig = px.bar(
            melted_ranking,
            x="Ranking",
            y="Embedding model",
            barmode="stack",
            color="Metric",
            orientation="h",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="")
    fig.update_yaxes(tickfont=dict(size=10))
    # fig.for_each_trace(lambda t: t.update(name=t.name.split('=')[1]))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    return fig


# -----------------------------------------------------------------------------
# ------------------------------- Correlations --------------------------------
# -----------------------------------------------------------------------------


def calculate_ranking_correlations(
    rankings: pd.DataFrame, method="spearman"
) -> pd.DataFrame:
    """Calculate the correlation between the calculated rankings.

    It uses pandas corr() method to calculate the correlations.
    The method parameter documentarion was copied from the documentation of the pandas
    DataFrame.corr() method.
    To see the updated documentation, visit:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html


    Parameters
    ----------
    rankings : pd.DataFrame
        DataFrame that contains the calculated rankings.

    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior.
            .. version

    Returns
    -------
    pd.DataFrame
        A dataframe with the calculated correlations.
    """
    if not isinstance(rankings, pd.DataFrame):
        raise TypeError(
            "rankings parameter must be a pandas DataFrame result of having "
            "executed create_rankings. Given: {}".format(rankings)
        )

    correlation_matrix = rankings.corr(method=method)
    return correlation_matrix


def plot_ranking_correlations(correlation_matrix, title=""):
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            hoverongaps=False,
            zmin=0.0,
            zmax=1,
            colorscale="Darkmint",
        )
    )
    fig.update_layout(title=title, font=dict(color="#000000"))
    return fig


def load_test_model() -> WordEmbeddingModel:
    """Load a Word2vec subset to test metrics and debias methods.

    Returns
    -------
    WordEmbeddingModel
        The loaded model
    """
    from gensim.models import KeyedVectors

    # load dummy weat word vectors:

    resource_package = __name__
    resource_path = "/".join(("datasets", "data", "test_model.kv"))
    weat_w2v_path = pkg_resources.resource_filename(resource_package, resource_path)

    test_model = KeyedVectors.load(weat_w2v_path)
    return WordEmbeddingModel(test_model, "Test w2v model")


def print_doc_table(df):
    from tabulate import tabulate

    df_2 = df.reset_index()
    print(tabulate(df_2, headers=df_2.columns, tablefmt="rst", showindex=False))


def save_doc_image(fig, name):
    fig.write_image(f"./doc/images/{name}.png", width=1200, height=600, scale=3)
