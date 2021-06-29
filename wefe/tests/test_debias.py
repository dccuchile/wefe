"""Tests of Hard Debias debiasing method."""
import pytest
from gensim.models.keyedvectors import KeyedVectors

from wefe.datasets import fetch_debiaswe, load_weat, fetch_debias_multiclass
from wefe.debias.hard_debias import HardDebias
from wefe.debias.multiclass_hard_debias import MulticlassHardDebias
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics import WEAT
from wefe.query import Query


@pytest.fixture
def model() -> WordEmbeddingModel:
    """Load a subset of Word2vec as a testing model.

    Returns
    -------
    WordEmbeddingModel
        The loaded testing model.
    """
    w2v = KeyedVectors.load("./wefe/tests/w2v_test.kv")
    return WordEmbeddingModel(w2v, "word2vec")


def test_hard_debias_class(model):

    debiaswe_wordsets = fetch_debiaswe()

    definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    equalize_pairs = debiaswe_wordsets["equalize_pairs"]
    gender_specific = debiaswe_wordsets["gender_specific"]

    hd = HardDebias()
    hd.fit(
        model,
        definitional_pairs,
        equalize_pairs=equalize_pairs,
        criterion_name="gender",
    )

    # -------------
    # Gender Debias
    gender_debiased_w2v = hd.transform(model, ignore=gender_specific, copy=True,)

    weat_word_set = load_weat()
    weat = WEAT()
    query = Query(
        [weat_word_set["male_names"], weat_word_set["female_names"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )
    biased_results = weat.run_query(query, model)
    debiased_results = weat.run_query(query, gender_debiased_w2v)

    assert debiased_results["weat"] < biased_results["weat"]


def test_multiclass_hard_debias_class(model):

    multiclass_debias_wordsets = fetch_debias_multiclass()
    weat_word_set = load_weat()
    weat = WEAT()

    # -------------
    # Gender Debias

    gender_definitional_sets = multiclass_debias_wordsets["gender_definitional_sets"]
    gender_equalize_sets = list(
        multiclass_debias_wordsets["gender_analogy_templates"].values()
    )

    mhd = MulticlassHardDebias()
    mhd.fit(
        model=model,
        definitional_sets=gender_definitional_sets,
        equalize_sets=gender_equalize_sets,
        criterion_name="gender",
    )

    gender_debiased_w2v = mhd.transform(model, copy=True)

    gender_query = Query(
        [weat_word_set["male_names"], weat_word_set["female_names"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )
    biased_results = weat.run_query(gender_query, model)
    debiased_results = weat.run_query(gender_query, gender_debiased_w2v)

    assert debiased_results["weat"] < biased_results["weat"]

    # ----------------
    # Ethnicity Debias

    ethnicity_definitional_sets = multiclass_debias_wordsets["race_definitional_sets"]
    ethnicity_equalize_sets = list(
        multiclass_debias_wordsets["race_analogy_templates"].values()
    )

    mhd = MulticlassHardDebias()
    mhd.fit(
        model=model,
        definitional_sets=ethnicity_definitional_sets,
        equalize_sets=ethnicity_equalize_sets,
        criterion_name="ethnicity",
    )

    ethnicity_debiased_w2v = mhd.transform(model, copy=True)

    ethnicity_query = Query(
        [
            weat_word_set["european_american_names_5"],
            weat_word_set["african_american_names_5"],
        ],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["European American Names", "African American Names"],
        ["Pleasant", "Unpleasant"],
    )
    biased_results = weat.run_query(ethnicity_query, model)
    debiased_results = weat.run_query(ethnicity_query, ethnicity_debiased_w2v)

    assert debiased_results["weat"] < biased_results["weat"]
