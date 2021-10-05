import gensim
import numpy as np
import pandas as pd
import pytest
import semantic_version
from gensim.models import KeyedVectors
from wefe.datasets import load_weat
from wefe.metrics import RND, WEAT
from wefe.query import Query
from wefe.utils import (
    calculate_ranking_correlations,
    create_ranking,
    load_test_model,
    run_queries,
)
from wefe.word_embedding_model import WordEmbeddingModel

gensim_version = semantic_version.Version.coerce(gensim.__version__)


def test_load_weat_w2v():
    test_model = load_test_model()
    assert isinstance(test_model, WordEmbeddingModel)
    assert isinstance(test_model.wv, KeyedVectors)

    NUM_OF_EMBEDDINGS_IN_TEST_MODEL = 13013
    if gensim_version.major >= 4:
        assert len(test_model.wv) == NUM_OF_EMBEDDINGS_IN_TEST_MODEL
        for word in test_model.wv.key_to_index:
            assert isinstance(test_model[word], np.ndarray)
    else:
        assert len(test_model.vocab.keys()) == NUM_OF_EMBEDDINGS_IN_TEST_MODEL
        for word in test_model.vocab:
            assert isinstance(test_model[word], np.ndarray)


@pytest.fixture
def queries_and_models():
    word_sets = load_weat()

    # Create gender queries
    gender_query_1 = Query(
        [word_sets["male_terms"], word_sets["female_terms"]],
        [word_sets["career"], word_sets["family"]],
        ["Male terms", "Female terms"],
        ["Career", "Family"],
    )
    gender_query_2 = Query(
        [word_sets["male_terms"], word_sets["female_terms"]],
        [word_sets["science"], word_sets["arts"]],
        ["Male terms", "Female terms"],
        ["Science", "Arts"],
    )
    gender_query_3 = Query(
        [word_sets["male_terms"], word_sets["female_terms"]],
        [word_sets["math"], word_sets["arts_2"]],
        ["Male terms", "Female terms"],
        ["Math", "Arts"],
    )

    # Create ethnicity queries
    test_query_1 = Query(
        [word_sets["insects"], word_sets["flowers"]],
        [word_sets["pleasant_5"], word_sets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )

    test_query_2 = Query(
        [word_sets["weapons"], word_sets["instruments"]],
        [word_sets["pleasant_5"], word_sets["unpleasant_5"]],
        ["Instruments", "Weapons"],
        ["Pleasant", "Unpleasant"],
    )

    gender_queries = [gender_query_1, gender_query_2, gender_query_3]
    negative_test_queries = [test_query_1, test_query_2]

    test_model = load_test_model()
    dummy_model_1 = test_model.wv
    dummy_model_2 = test_model.wv
    dummy_model_3 = test_model.wv

    models = [
        WordEmbeddingModel(dummy_model_1, "dummy_model_1"),
        WordEmbeddingModel(dummy_model_2, "dummy_model_2"),
        WordEmbeddingModel(dummy_model_3, "dummy_model_3"),
    ]
    return gender_queries, negative_test_queries, models


def test_run_query_input_validation(queries_and_models):

    # -----------------------------------------------------------------
    # Input checks
    # -----------------------------------------------------------------

    # Load the inputs of the fixture
    gender_queries, _, models = queries_and_models

    with pytest.raises(
        TypeError, match="queries parameter must be a list or a numpy array*"
    ):
        run_queries(WEAT, None, None)

    with pytest.raises(
        Exception, match="queries list must have at least one query instance*"
    ):
        run_queries(WEAT, [], None)

    with pytest.raises(TypeError, match="item on index 0 must be a Query instance*"):
        run_queries(WEAT, [None, None], None)

    with pytest.raises(TypeError, match="item on index 3 must be a Query instance*"):
        run_queries(WEAT, gender_queries + [None], None)

    with pytest.raises(
        TypeError,
        match="word_embeddings_models parameter must be a list or a " "numpy array*",
    ):
        run_queries(WEAT, gender_queries, None)

    with pytest.raises(
        Exception,
        match="word_embeddings_models parameter must be a non empty list "
        "or numpy array*",
    ):
        run_queries(WEAT, gender_queries, [])

    with pytest.raises(
        TypeError, match="item on index 0 must be a WordEmbeddingModel instance*"
    ):
        run_queries(WEAT, gender_queries, [None])

    with pytest.raises(
        TypeError, match="item on index 3 must be a WordEmbeddingModel instance*"
    ):
        run_queries(WEAT, gender_queries, models + [None])

    with pytest.raises(
        TypeError,
        match="When queries_set_name parameter is provided, it must be a "
        "non-empty string*",
    ):
        run_queries(WEAT, gender_queries, models, queries_set_name=None)

    with pytest.raises(
        TypeError,
        match="When queries_set_name parameter is provided, it must be a "
        "non-empty string*",
    ):
        run_queries(WEAT, gender_queries, models, queries_set_name="")

    with pytest.raises(
        TypeError,
        match="run_experiment_params must be a dict with a params" " for the metric*",
    ):
        run_queries(WEAT, gender_queries, models, metric_params=None)

    with pytest.raises(
        Exception, match="aggregate_results parameter must be a bool value*"
    ):
        run_queries(WEAT, gender_queries, models, aggregate_results=None)

    with pytest.raises(
        Exception,
        match="aggregation_function must be one of 'sum',"
        "abs_sum', 'avg', 'abs_avg' or a callable.*",
    ):
        run_queries(WEAT, gender_queries, models, aggregation_function=None)

    with pytest.raises(
        Exception,
        match="aggregation_function must be one of 'sum',"
        "abs_sum', 'avg', 'abs_avg' or a callable.*",
    ):
        run_queries(WEAT, gender_queries, models, aggregation_function="hello")

    with pytest.raises(
        Exception, match="return_only_aggregation param must be boolean.*"
    ):
        run_queries(WEAT, gender_queries, models, return_only_aggregation=None)


def test_run_queries(queries_and_models):
    def check_results_types(results, only_negative=False):
        for row in results.values:
            for value in row:
                assert isinstance(value, np.float_)
                if only_negative:
                    assert value <= 0

    # -----------------------------------------------------------------
    # Basic run_queries execution
    # -----------------------------------------------------------------

    # Load the inputs of the fixture
    gender_queries, negative_test_queries, models = queries_and_models

    results = run_queries(metric=WEAT, queries=gender_queries, models=models,)

    assert isinstance(results, pd.DataFrame)
    assert results.shape == (3, 3)

    # Check cols
    expected_cols = [
        "Male terms and Female terms wrt Career and Family",
        "Male terms and Female terms wrt Science and Arts",
        "Male terms and Female terms wrt Math and Arts",
    ]

    for given_col, expected_col in zip(results.columns, expected_cols):
        assert given_col == expected_col

    # Check index
    expected_index = ["dummy_model_1", "dummy_model_2", "dummy_model_3"]

    for given_idx, expected_idx in zip(results.index, expected_index):
        assert given_idx, expected_idx

    # Check values
    check_results_types(results)

    results = run_queries(WEAT, negative_test_queries, models)
    check_results_types(results, only_negative=True)

    # -----------------------------------------------------------------
    # run_queries with params execution
    # -----------------------------------------------------------------

    # lost_vocabulary_threshold...
    word_sets = load_weat()
    dummy_query_1 = Query(
        [["bla", "ble", "bli"], word_sets["insects"]],
        [word_sets["pleasant_9"], word_sets["unpleasant_9"]],
    )

    results = run_queries(
        WEAT, gender_queries + [dummy_query_1], models, lost_vocabulary_threshold=0.1
    )
    assert results.shape == (3, 4)
    assert results.isnull().any().any()
    check_results_types(results)

    # metric param...
    results = run_queries(
        WEAT, gender_queries, models, metric_params={"return_effect_size": True}
    )
    assert results.shape == (3, 3)
    check_results_types(results)

    # -----------------------------------------------------------------
    # run_queries with aggregation params execution
    # -----------------------------------------------------------------

    # include_average_by_embedding..
    results = run_queries(WEAT, gender_queries, models, aggregate_results=True)
    assert results.shape == (3, 4)
    check_results_types(results)

    # avg
    results = run_queries(
        WEAT,
        negative_test_queries,
        models,
        aggregate_results=True,
        aggregation_function="avg",
    )
    assert results.shape == (3, 3)
    check_results_types(results)
    agg = results.values[:, 2]
    values = results.values[:, 0:2]
    calc_agg = np.mean(values, axis=1)
    assert np.array_equal(agg, calc_agg)

    # abs avg
    results = run_queries(
        WEAT,
        negative_test_queries,
        models,
        aggregate_results=True,
        aggregation_function="abs_avg",
    )
    assert results.shape == (3, 3)
    check_results_types(results)
    agg = results.values[:, 2]
    values = results.values[:, 0:2]
    calc_agg = np.mean(np.abs(values), axis=1)
    assert np.array_equal(agg, calc_agg)

    # sum
    results = run_queries(
        WEAT,
        negative_test_queries,
        models,
        aggregate_results=True,
        aggregation_function="sum",
    )
    assert results.shape == (3, 3)
    check_results_types(results)
    agg = results.values[:, 2]
    values = results.values[:, 0:2]
    calc_agg = np.sum(values, axis=1)
    assert np.array_equal(agg, calc_agg)

    # abs_sum
    results = run_queries(
        WEAT,
        negative_test_queries,
        models,
        aggregate_results=True,
        aggregation_function="abs_sum",
    )
    assert results.shape == (3, 3)
    check_results_types(results)
    agg = results.values[:, 2]
    values = results.values[:, 0:2]
    calc_agg = np.sum(np.abs(values), axis=1)
    assert np.array_equal(agg, calc_agg)

    # custom agg function
    results = run_queries(
        WEAT,
        negative_test_queries,
        models,
        aggregate_results=True,
        aggregation_function=lambda df: -df.abs().mean(1),
    )
    assert results.shape == (3, 3)
    check_results_types(results, only_negative=True)
    agg = results.values[:, 2]
    values = results.values[:, 0:2]
    calc_agg = -np.mean(np.abs(values), axis=1)
    assert np.array_equal(agg, calc_agg)

    # return only aggregation without query name
    results = run_queries(
        WEAT,
        gender_queries,
        models,
        aggregate_results=True,
        aggregation_function="abs_avg",
        return_only_aggregation=True,
    )
    assert results.shape == (3, 1)
    check_results_types(results)
    assert (
        results.columns[-1] == "WEAT: Unnamed queries set average of abs values score"
    )

    # return only aggregation without query name
    results = run_queries(
        WEAT,
        gender_queries,
        models,
        aggregate_results=True,
        aggregation_function="abs_avg",
        queries_set_name="Gender queries",
        return_only_aggregation=True,
    )
    assert results.shape == (3, 1)
    check_results_types(results)
    assert results.columns[-1] == "WEAT: Gender queries average of abs values score"

    # return aggregation with query name
    results = run_queries(
        WEAT,
        gender_queries,
        models,
        aggregate_results=True,
        aggregation_function="abs_avg",
        queries_set_name="Gender queries",
        return_only_aggregation=False,
    )
    assert results.shape == (3, 4)
    check_results_types(results)
    assert results.columns[-1] == "WEAT: Gender queries average of abs values score"
    # -----------------------------------------------------------------
    # run_queries with generate subqueries params execution
    # -----------------------------------------------------------------

    # with this option, the gender queries must be divided in RND template
    # (2,1). with one query replicated (arts), the remaining are only 5.
    results = run_queries(RND, gender_queries, models, generate_subqueries=True)
    assert results.shape == (3, 5)
    check_results_types(results)

    # -----------------------------------------------------------------
    # run_queries full params execution
    # -----------------------------------------------------------------
    results = run_queries(
        RND,
        gender_queries,
        models,
        queries_set_name="Gender queries",
        generate_subqueries=True,
        aggregate_results=True,
        aggregation_function="abs_avg",
        return_only_aggregation=False,
        metric_params={"distance_type": "cos"},
    )
    assert results.shape == (3, 6)
    check_results_types(results)
    assert results.columns[-1] == "RND: Gender queries average of abs values score"


def test_rank_results(queries_and_models):

    gender_queries, negative_test_queries, models = queries_and_models

    results_gender = run_queries(
        WEAT,
        gender_queries,
        models,
        queries_set_name="Gender Queries",
        aggregate_results=True,
    )
    results_negative = run_queries(
        WEAT,
        negative_test_queries,
        models,
        queries_set_name="Negative Test Queries",
        aggregate_results=True,
    )

    results_gender_rnd = run_queries(
        RND,
        gender_queries,
        models,
        queries_set_name="Gender Queries",
        generate_subqueries=True,
        aggregate_results=True,
    )

    with pytest.raises(
        TypeError,
        match="All elements of results_dataframes must be a pandas "
        "Dataframe instance*",
    ):
        create_ranking([None, results_gender, results_negative])

    with pytest.raises(
        TypeError,
        match="All elements of results_dataframes must be a pandas "
        "Dataframe instance*",
    ):
        create_ranking([results_gender, results_negative, 2])

    ranking = create_ranking([results_gender, results_negative, results_gender_rnd])

    expected_ranking = pd.DataFrame(
        {
            "WEAT: Gender Queries average of abs values score": {
                "dummy_model_1": 1.0,
                "dummy_model_2": 2.0,
                "dummy_model_3": 3.0,
            },
            "WEAT: Negative Test Queries average of abs values score": {
                "dummy_model_1": 1.0,
                "dummy_model_2": 2.0,
                "dummy_model_3": 3.0,
            },
            "RND: Gender Queries average of abs values score": {
                "dummy_model_1": 1.0,
                "dummy_model_2": 2.0,
                "dummy_model_3": 3.0,
            },
        }
    )
    assert ranking.shape == (3, 3)
    assert expected_ranking.equals(ranking)

    for row in ranking.values:
        for val in row:
            assert val <= 3 and val >= 1


def test_correlations(queries_and_models):

    gender_queries, negative_test_queries, models = queries_and_models
    results_gender = run_queries(
        WEAT,
        gender_queries,
        models,
        queries_set_name="Gender Queries",
        aggregate_results=True,
    )
    results_negative = run_queries(
        WEAT,
        negative_test_queries,
        models,
        queries_set_name="Negative Test Queries",
        aggregate_results=True,
        return_only_aggregation=True,
    )

    results_gender_rnd = run_queries(
        RND,
        gender_queries,
        models,
        queries_set_name="Gender Queries",
        generate_subqueries=True,
        aggregate_results=True,
    )

    ranking = create_ranking([results_gender, results_negative, results_gender_rnd])
    assert ranking.shape == (3, 3)

    correlations = calculate_ranking_correlations(ranking)
    assert correlations.shape == (3, 3)
    assert np.array_equal(
        correlations.columns.values,
        np.array(
            [
                "WEAT: Gender Queries average of abs values score",
                "WEAT: Negative Test Queries average of abs values score",
                "RND: Gender Queries average of abs values score",
            ]
        ),
    )
    assert np.array_equal(
        correlations.index.values,
        np.array(
            [
                "WEAT: Gender Queries average of abs values score",
                "WEAT: Negative Test Queries average of abs values score",
                "RND: Gender Queries average of abs values score",
            ]
        ),
    )

