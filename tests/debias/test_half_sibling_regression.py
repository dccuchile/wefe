"""Half Sibling Regression (HSR) test set."""
from typing import Dict, List

import numpy as np
import pytest
from wefe.debias.half_sibling_regression import HalfSiblingRegression
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_half_sibling_checks(model):
    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got .*",
    ):
        HalfSiblingRegression(verbose=1)


def test_half_sibling_regression_class(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    gender_specific: List[str],
):
    weat = WEAT()

    hsr = HalfSiblingRegression(criterion_name="gender",)
    hsr.fit(model, definitional_words=gender_specific)

    gender_debiased_w2v = hsr.transform(model, copy=True)

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]


def test_half_sibling_regression_target_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    control_query_1: Query,
    gender_specific: List[str],
    weat_wordsets: Dict[str, List[str]],
):
    weat = WEAT()

    hsr = HalfSiblingRegression(criterion_name="gender",)

    attribute_words = weat_wordsets["career"] + weat_wordsets["family"]
    attribute_words.remove("family")
    attribute_words.remove("executive")

    gender_debiased_w2v = hsr.fit(model, definitional_words=gender_specific).transform(
        model, target=attribute_words, copy=True
    )

    # test gender query 1, debias was not applied to any word
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert np.isclose(abs(debiased_results["weat"]), abs(biased_results["weat"]))

    # test gender query 2, debias was applied to the target and attribute words.
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])

    # test control_query_1 (flowers vs insects wrt pleasant vs unpleasant), debias
    # was not applied to their target and attribute words
    biased_results = weat.run_query(control_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        control_query_1, gender_debiased_w2v, normalize=True
    )
    assert np.isclose(debiased_results["weat"], biased_results["weat"])


def test_half_sibling_regression_ignore_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    gender_specific: List[str],
    weat_wordsets: Dict[str, List[str]],
):
    weat = WEAT()

    hsr = HalfSiblingRegression(verbose=True, criterion_name="gender",)

    targets = weat_wordsets["male_names"] + weat_wordsets["female_names"]
    attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]
    ignore = targets + attributes

    gender_debiased_w2v = hsr.fit(model, definitional_words=gender_specific).transform(
        model, ignore=ignore, copy=True
    )

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.

    # test gender query 1,none of their words were debiased.
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )

    assert np.isclose(debiased_results["weat"], biased_results["weat"])

    # test gender query 2, debias was not applied to their target words
    # (career vs family), but to its attributes.
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])


def test_double_hard_debias_copy_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    gender_specific: List[str],
):
    weat = WEAT()
    # Since we will mutate the original model in the test, we calculate WEAT scores
    # before debiasing with the original model.
    biased_results_q1 = weat.run_query(gender_query_1, model, normalize=True)
    biased_results_q2 = weat.run_query(gender_query_2, model, normalize=True)

    # Test inplace (copy = False)
    hsr = HalfSiblingRegression(verbose=True, criterion_name="gender",)
    hsr.fit(model, definitional_words=gender_specific)

    gender_debiased_w2v = hsr.transform(model, ignore=gender_specific, copy=False)

    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name

    # tests with WEAT
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results_q1["weat"])

    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results_q2["weat"])


def test_verbose(
    model: WordEmbeddingModel, gender_specific: List[str], capsys,
):

    # -----------------------------------------------------------------
    # Test verbose
    hsr = HalfSiblingRegression(verbose=True)
    gender_debiased_w2v = hsr.fit(model, definitional_words=gender_specific).transform(
        model, copy=True
    )

    out = capsys.readouterr().out
    assert "Computing the weight matrix." in out
    assert "Computing bias information" in out
    assert f"Executing Half Sibling Debias on {model.name}" in out
    assert "Copy argument is True. Transform will attempt to create a copy" in out
    assert "Subtracting bias information." in out
    assert "Updating debiased vectors" in out
    assert "Done!" in out

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_debiased"

    # -----------------------------------------------------------------
    # Test inplace (copy = False)
    hsr = HalfSiblingRegression(criterion_name="gender",)
    hsr.fit(model, definitional_words=gender_specific)

    gender_debiased_w2v = hsr.transform(model, copy=False)
    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name
