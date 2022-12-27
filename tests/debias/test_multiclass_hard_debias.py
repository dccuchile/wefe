"""Multiclass Hard Debias (MHD) test sets."""
from typing import Dict, List

import numpy as np
import pytest

from wefe.debias.multiclass_hard_debias import MulticlassHardDebias
from wefe.metrics import MAC, WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_multiclass_hard_debias_param_checks(
    model: WordEmbeddingModel,
    definitional_pairs: List[List[str]],
):

    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1', 'word2', 'word3'\]\) "
            r"has more words than the other definitional sets: "
            r"got 3 words, expected 2\."
        ),
    ):
        MulticlassHardDebias().fit(
            model,
            definitional_sets=definitional_pairs + [["word1", "word2", "word3"]],
            equalize_sets=definitional_pairs,
        )
    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1'\]\) has less words "
            r"than the other definitional sets: got 1 words, expected 2\."
        ),
    ):
        MulticlassHardDebias().fit(
            model,
            definitional_sets=definitional_pairs + [["word1"]],
            equalize_sets=definitional_pairs,
        )


def test_multiclass_hard_debias_with_gender(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    gender_query_3: Query,
    mhd_gender_definitional_sets: List[List[str]],
    mhd_gender_equalize_sets: List[List[str]],
):
    weat = WEAT()

    # note that the original paper applies the debias over all words
    mhd = MulticlassHardDebias(criterion_name="gender")
    mhd.fit(
        model=model,
        definitional_sets=mhd_gender_definitional_sets,
        equalize_sets=mhd_gender_equalize_sets,
    )
    gender_debiased_w2v = mhd.transform(model, copy=True)

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

    # tests with WEAT (close to 0 is better)
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

    # test with MAC (from the original repository)
    biased_results_2 = MAC().run_query(gender_query_3, model, normalize=True)
    debiased_results_2 = MAC().run_query(
        gender_query_3, gender_debiased_w2v, normalize=True
    )
    # in this case, closer to one (higher) is better:
    assert debiased_results_2["mac"] > biased_results_2["mac"]


def test_multiclass_hard_debias_with_ethnicity(
    model: WordEmbeddingModel,
    ethnicity_query_1: Query,
    mhd_ethnicity_definitional_sets: List[List[str]],
    mhd_ethnicity_equalize_sets: List[List[str]],
):
    weat = WEAT()

    mhd = MulticlassHardDebias(verbose=True, criterion_name="ethnicity")
    mhd.fit(
        model=model,
        definitional_sets=mhd_ethnicity_definitional_sets,
        equalize_sets=mhd_ethnicity_equalize_sets,
    )
    ethnicity_debiased_w2v = mhd.transform(model, copy=True)

    assert model.name == "test_w2v"
    assert ethnicity_debiased_w2v.name == "test_w2v_ethnicity_debiased"

    # test with weat
    biased_results = weat.run_query(ethnicity_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        ethnicity_query_1, ethnicity_debiased_w2v, normalize=True
    )

    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])


def test_multiclass_hard_debias_target_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    control_query_1: Query,
    mhd_gender_definitional_sets: List[List[str]],
    mhd_gender_equalize_sets: List[List[str]],
    weat_wordsets: Dict[str, List[str]],
):
    weat = WEAT()
    # -----------------------------------------------------------------
    # Test target param

    mhd = MulticlassHardDebias()

    attribute_words = weat_wordsets["career"] + weat_wordsets["family"]
    attribute_words.remove("family")
    attribute_words.remove("executive")

    gender_debiased_w2v = mhd.fit(
        model,
        definitional_sets=mhd_gender_definitional_sets,
        equalize_sets=mhd_gender_equalize_sets,
    ).transform(
        model,
        target=attribute_words,
        copy=True,
    )

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_debiased"

    # test gender query 1, debias was only applied to the target words
    # (in equalization step).
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])

    # test gender query 2, debias was applied to the target (in equalization
    # step) and attribute words (in neutralization step).
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])

    # test control_query_1 (flowers vs insects wrt pleasant vs unpleasant), debias
    # was not applied to their target (equalization) and attribute words
    # (neutralization).
    biased_results = weat.run_query(control_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        control_query_1, gender_debiased_w2v, normalize=True
    )
    assert np.isclose(debiased_results["weat"], biased_results["weat"])


def test_multiclass_hard_debias_ignore_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    control_query_1: Query,
    mhd_gender_definitional_sets: List[List[str]],
    mhd_gender_equalize_sets: List[List[str]],
    weat_wordsets: Dict[str, List[str]],
):
    weat = WEAT()
    # -----------------------------------------------------------------
    # Test ignore param
    mhd = MulticlassHardDebias(criterion_name="gender")

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.
    targets = weat_wordsets["male_names"] + weat_wordsets["female_names"]
    attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]
    ignore = targets + attributes

    gender_debiased_w2v = mhd.fit(
        model,
        definitional_sets=mhd_gender_definitional_sets,
        equalize_sets=mhd_gender_equalize_sets,
    ).transform(model, ignore=ignore, copy=True)

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


def test_multiclass_hard_debias_copy_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    mhd_gender_definitional_sets: List[List[str]],
    mhd_gender_equalize_sets: List[List[str]],
):
    weat = WEAT()
    # Since we will mutate the original model in the test, we calculate WEAT scores
    # before debiasing with the original model.
    biased_results_q1 = weat.run_query(gender_query_1, model, normalize=True)
    biased_results_q2 = weat.run_query(gender_query_2, model, normalize=True)

    # Test inplace (copy = False)
    mhd = MulticlassHardDebias(criterion_name="gender")
    mhd.fit(
        model=model,
        definitional_sets=mhd_gender_definitional_sets,
        equalize_sets=mhd_gender_equalize_sets,
    )

    gender_debiased_w2v = mhd.transform(model, copy=False)

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
