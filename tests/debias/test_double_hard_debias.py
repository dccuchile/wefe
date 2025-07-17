"""Double Hard Debias (DHD) test set."""

import numpy as np
import pytest

from wefe.debias.double_hard_debias import DoubleHardDebias
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_double_hard_debias_checks(
    model: WordEmbeddingModel, definitional_pairs: list[list[str]]
) -> None:
    with pytest.raises(
        TypeError,
        match=r"verbose should be a bool, got .*",
    ):
        DoubleHardDebias(verbose=1)

    with pytest.raises(
        TypeError,
        match=r"n_words should be int, got: .*",
    ):
        DoubleHardDebias(n_words=2.3)

    with pytest.raises(
        TypeError,
        match=r"n_components should be int, got: .*",
    ):
        DoubleHardDebias(n_components=2.3)
    with pytest.raises(
        TypeError,
        match=r"incremental_pca should be a bool, got .*",
    ):
        DoubleHardDebias(incremental_pca=1)

    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1', 'word2', 'word3'\]\) "
            r"has more words than allowed by Double Hard Debias: got 3 words, "
            r"expected 2\."
        ),
    ):
        DoubleHardDebias().fit(
            model,
            definitional_pairs=definitional_pairs + [["word1", "word2", "word3"]],
            bias_representation=["he", "she"],
        )
    with pytest.raises(
        ValueError,
        match=(
            r"The definitional set at position 10 \(\['word1'\]\) has less words "
            r"than allowed by Double Hard Debias: got 1 words, expected 2\."
        ),
    ):
        DoubleHardDebias().fit(
            model, definitional_pairs + [["word1"]], bias_representation=["he", "she"]
        )
    with pytest.raises(
        Exception,
        match=r"bias_representation words not in model",
    ):
        DoubleHardDebias().fit(
            model,
            definitional_pairs,
            bias_representation=["abcde123efg", "gfe321edcba"],
        )


def test_double_hard_debias(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    definitional_pairs: list[list[str]],
    gender_specific: list[str],
) -> None:
    weat = WEAT()

    dhd = DoubleHardDebias(
        criterion_name="gender",
    )
    dhd.fit(
        model, definitional_pairs=definitional_pairs, bias_representation=["he", "she"]
    )
    gender_debiased_w2v = dhd.transform(model, ignore=gender_specific)

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

    # tests with WEAT (close to 0 is better)
    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])

    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert abs(debiased_results["weat"]) < abs(biased_results["weat"])


def test_double_hard_debias_target_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    control_query_1: Query,
    definitional_pairs: list[list[str]],
    weat_wordsets: dict[str, list[str]],
) -> None:
    weat = WEAT()

    dhd = DoubleHardDebias(
        verbose=True,
        criterion_name="gender",
    )

    attribute_words = weat_wordsets["career"] + weat_wordsets["family"]
    attribute_words.remove("family")
    attribute_words.remove("executive")

    gender_debiased_w2v = dhd.fit(
        model, definitional_pairs=definitional_pairs, bias_representation=["he", "she"]
    ).transform(model, target=attribute_words, copy=True)

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
    definitional_pairs: list[list[str]],
    weat_wordsets: dict[str, list[str]],
) -> None:
    weat = WEAT()
    dhd = DoubleHardDebias(
        verbose=True,
        criterion_name="gender",
    )

    targets = weat_wordsets["male_names"] + weat_wordsets["female_names"]
    attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]
    ignore = targets + attributes

    gender_debiased_w2v = dhd.fit(
        model, definitional_pairs=definitional_pairs, bias_representation=["he", "she"]
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


def test_double_hard_debias_copy_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    definitional_pairs: list[list[str]],
    gender_specific: list[str],
) -> None:
    weat = WEAT()
    # Since we will mutate the original model in the test, we calculate WEAT scores
    # before debiasing with the original model.
    biased_results_q1 = weat.run_query(gender_query_1, model, normalize=True)
    biased_results_q2 = weat.run_query(gender_query_2, model, normalize=True)

    # Test inplace (copy = False)
    dhd = DoubleHardDebias(
        criterion_name="gender",
    )
    dhd.fit(
        model, definitional_pairs=definitional_pairs, bias_representation=["he", "she"]
    )

    gender_debiased_w2v = dhd.transform(model, ignore=gender_specific, copy=False)

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


def test_multiclass_hard_debias_verbose(
    model: WordEmbeddingModel,
    definitional_pairs: list[list[str]],
    gender_specific: list[str],
    capsys,
) -> None:
    # -----------------------------------------------------------------
    # Test verbose
    dhd = DoubleHardDebias(verbose=True)
    gender_debiased_w2v = dhd.fit(
        model, definitional_pairs, bias_representation=["he", "she"]
    ).transform(model, ignore=gender_specific, copy=True)
    out = capsys.readouterr().out
    assert "Obtaining definitional pairs." in out
    assert "PCA variance explained:" in out
    assert "Identifying the bias subspace" in out
    assert "Obtaining definitional pairs." in out
    assert f"Executing Double Hard Debias on {model.name}" in out
    assert "Identifying the bias subspace." in out
    assert "Obtaining principal components" in out
    assert "Obtaining words to apply debias" in out
    assert "Searching component to debias" in out
    assert "Copy argument is True. Transform will attempt to create a copy" in out
    assert "Executing debias" in out

    dhd = DoubleHardDebias(
        criterion_name="gender",
    )
    dhd.fit(
        model, definitional_pairs=definitional_pairs, bias_representation=["he", "she"]
    )

    gender_debiased_w2v = dhd.transform(model, ignore=gender_specific, copy=False)
    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name
