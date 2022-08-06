"""Set of Tests for mitigation methods."""
from typing import Dict, List

import numpy as np
import pytest
from wefe.debias.hard_debias import HardDebias
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_hard_debias_param_checks(
    model: WordEmbeddingModel, definitional_pairs: List[List[str]],
):

    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got .*",
    ):
        HardDebias(verbose=1)

    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1', 'word2', 'word3'\]\) "
            r"has more words than allowed by Hard Debias: got 3 words, expected 2\."
        ),
    ):
        HardDebias().fit(
            model, definitional_pairs + [["word1", "word2", "word3"]],
        )
    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1'\]\) has less words "
            r"than allowed by Hard Debias: got 1 words, expected 2\."
        ),
    ):
        HardDebias().fit(
            model, definitional_pairs + [["word1"]],
        )


def test_hard_debias_class(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    definitional_pairs: List[str],
    equalize_pairs: List[List[str]],
    gender_specific: List[List[str]],
):

    weat = WEAT()

    # -----------------------------------------------------------------
    # Gender Debias
    hd = HardDebias(criterion_name="gender")
    hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    )

    gender_debiased_w2v = hd.transform(model, ignore=gender_specific, copy=True)

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

    # check gender query 1 in original and debiased model.
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]

    # check gender query 1 in original and debiased model.
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]


def test_hard_debias_target_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    weat_wordsets: Dict[str, List[str]],
    definitional_pairs: List[List[str]],
    equalize_pairs: List[List[str]],
):
    weat = WEAT()
    hd = HardDebias(verbose=True, criterion_name="gender",)

    target = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]

    gender_debiased_w2v = hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    ).transform(model, target=target, copy=True)

    # test gender query 1, debias was applied to their attribute words.
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]

    # test gender query 2, debias was not applied to their attribute words.
    # however, in the equalization stage the targets were modified, so there should be
    # significant differences.
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]


def test_hard_debias_ignore_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    weat_wordsets: Dict[str, List[str]],
    definitional_pairs: List[List[str]],
    equalize_pairs: List[List[str]],
    gender_specific: List[str],
):
    weat = WEAT()
    hd = HardDebias(verbose=True, criterion_name="gender")

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.
    targets = weat_wordsets["male_names"] + weat_wordsets["female_names"]
    query_1_attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]
    gender_debiased_w2v = hd.fit(
        model, definitional_pairs, equalize_pairs=equalize_pairs,
    ).transform(model, ignore=gender_specific + targets + query_1_attributes, copy=True)

    # test gender query 1, debias was not applied to their attribute words.
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )

    assert np.isclose(debiased_results["weat"] - biased_results["weat"], 0)

    # test gender query 2, debias was applied to their attribute words
    # (by not specifying them in the ignore parameter.).
    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] - biased_results["weat"] != 0.000000


def test_hard_debias_verbose_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    definitional_pairs: List[str],
    equalize_pairs: List[str],
    gender_specific: List[str],
    capsys,
):
    weat = WEAT()
    # -----------------------------------------------------------------
    # Test verbose
    hd = HardDebias(verbose=True)
    gender_debiased_w2v = hd.fit(
        model, definitional_pairs, equalize_pairs=equalize_pairs,
    ).transform(model, ignore=gender_specific, copy=True)

    out = capsys.readouterr().out
    assert "Obtaining definitional pairs." in out
    assert "PCA variance explained:" in out
    assert "Identifying the bias subspace" in out
    assert "Obtaining equalize pairs candidates by creating" in out
    assert "Obtaining equalize pairs" in out
    assert f"Executing Hard Debias on {model.name}" in out
    assert "Copy argument is True. Transform will attempt to create a copy" in out
    assert "Normalizing embeddings" in out
    assert "Neutralizing embeddings" in out
    assert "Equalizing embeddings" in out
    assert "Done!" in out

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_debiased"

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


def test_hard_debias_copy_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    definitional_pairs: List[str],
    equalize_pairs: List[str],
    gender_specific: List[str],
):
    weat = WEAT()
    # Since we will mutate the original model in the test, we calculate WEAT scores
    # before debiasing with the original model.
    biased_results_q1 = weat.run_query(gender_query_1, model, normalize=True)
    biased_results_q2 = weat.run_query(gender_query_2, model, normalize=True)

    hd = HardDebias(criterion_name="gender",)
    hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    )

    gender_debiased_w2v = hd.transform(model, ignore=gender_specific, copy=False)
    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name

    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results_q1["weat"]

    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results_q2["weat"]
