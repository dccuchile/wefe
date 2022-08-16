"""Set of Tests for mitigation methods."""
import pytest
from wefe.datasets import fetch_debiaswe, load_weat
from wefe.debias.half_sibling_regression import HalfSiblingRegression
from wefe.metrics import WEAT
from wefe.query import Query


def test_half_sibling_checks(model):
    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got .*",
    ):
        HalfSiblingRegression(verbose=1)


def test_half_sibling_regression_class(model, capsys):

    # -----------------------------------------------------------------
    # Queries
    weat_wordset = load_weat()
    weat = WEAT()
    query_1 = Query(
        [weat_wordset["male_names"], weat_wordset["female_names"]],
        [weat_wordset["pleasant_5"], weat_wordset["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )
    query_2 = Query(
        [weat_wordset["male_names"], weat_wordset["female_names"]],
        [weat_wordset["career"], weat_wordset["family"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )

    debiaswe_wordsets = fetch_debiaswe()

    gender_specific = debiaswe_wordsets["gender_specific"]

    # -----------------------------------------------------------------
    # Gender Debias
    hsr = HalfSiblingRegression(criterion_name="gender",)
    hsr.fit(model, bias_definitional_words=gender_specific)

    gender_debiased_w2v = hsr.transform(model, copy=True)

    assert model.name == "test_w2v"
    assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(query_2, model, normalize=True)
    debiased_results = weat.run_query(query_2, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    # -----------------------------------------------------------------
    # Test target param
    hsr = HalfSiblingRegression(verbose=True, criterion_name="gender",)

    attributes = weat_wordset["pleasant_5"] + weat_wordset["unpleasant_5"]

    gender_debiased_w2v = hsr.fit(
        model, bias_definitional_words=gender_specific
    ).transform(model, target=attributes, copy=True)

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(query_2, model, normalize=True)
    debiased_results = weat.run_query(query_2, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] - biased_results["weat"] < 0.000000

    # -----------------------------------------------------------------
    # Test ignore param
    hsr = HalfSiblingRegression(verbose=True, criterion_name="gender",)

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.
    targets = weat_wordset["male_names"] + weat_wordset["female_names"]
    attributes = weat_wordset["pleasant_5"] + weat_wordset["unpleasant_5"]
    gender_debiased_w2v = hsr.fit(
        model, bias_definitional_words=gender_specific
    ).transform(model, ignore=gender_specific + targets + attributes, copy=True)

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)

    assert debiased_results["weat"] - biased_results["weat"] < 0.0000001

    # -----------------------------------------------------------------
    # Test verbose
    hsr = HalfSiblingRegression(verbose=True)
    gender_debiased_w2v = hsr.fit(
        model, bias_definitional_words=gender_specific
    ).transform(model, copy=True)

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
    hsr.fit(model, bias_definitional_words=gender_specific)

    gender_debiased_w2v = hsr.transform(model, copy=False)
    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name
