"""Tests of Hard Debias debiasing method."""
import pytest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from wefe.datasets import fetch_debiaswe, load_weat, fetch_debias_multiclass
from wefe.debias.base_debias import BaseDebias
from wefe.debias.hard_debias import HardDebias
from wefe.debias.multiclass_hard_debias import MulticlassHardDebias
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics import WEAT, MAC
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


def test_base_debias(model):

    bd = BaseDebias()
    with pytest.raises(NotImplementedError,):
        bd.fit(None)
    with pytest.raises(NotImplementedError,):
        bd.transform(None)
    with pytest.raises(NotImplementedError,):
        bd.fit_transform(None)

    debiaswe_wordsets = fetch_debiaswe()
    gender_specific = debiaswe_wordsets["gender_specific"]

    # type checking function
    with pytest.raises(
        TypeError, match=r"model should be a WordEmbeddingModel instance, got .*",
    ):
        bd._check_transform_args(None)

    with pytest.raises(
        TypeError, match=r"target should be None or a list of strings, got .*",
    ):
        bd._check_transform_args(model, target=1)
    with pytest.raises(
        TypeError, match=r"All elements in target should be strings, .*",
    ):
        bd._check_transform_args(model, target=gender_specific + [10])

    with pytest.raises(
        TypeError, match=r"ignore should be None or a list of strings, got .*",
    ):
        bd._check_transform_args(model, ignore=1)
    with pytest.raises(
        TypeError, match=r"All elements in ignore should be strings, .*",
    ):
        bd._check_transform_args(model, ignore=gender_specific + [10])

    with pytest.raises(
        TypeError, match=r"copy should be a bool, got .*",
    ):
        bd._check_transform_args(model, copy=None)

    assert (
        bd._check_transform_args(
            model, target=["word1", "word2"], ignore=gender_specific, copy=False
        )
        is None
    )


def test_hard_debias_checks(model):
    debiaswe_wordsets = fetch_debiaswe()

    definitional_pairs = debiaswe_wordsets["definitional_pairs"]

    with pytest.raises(
        TypeError, match=r"verbose should be a bool, got .*",
    ):
        HardDebias(verbose=1)

    with pytest.raises(
        ValueError,
        match=r"The definitional pair at position 10 \(\['word1', 'word2', 'word3'\]\) has more words than allowed by Hard Debias: got 3 words, expected 2\.",
    ):
        HardDebias().fit(
            model, definitional_pairs + [["word1", "word2", "word3"]],
        )
    with pytest.raises(
        ValueError,
        match=r"The definitional pair at position 10 \(\['word1'\]\) has less words than allowed by Hard Debias: got 1 words, expected 2\.",
    ):
        HardDebias().fit(
            model, definitional_pairs + [["word1"]],
        )


def test_hard_debias_class(model, capsys):

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

    definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    equalize_pairs = debiaswe_wordsets["equalize_pairs"]
    gender_specific = debiaswe_wordsets["gender_specific"]

    # -----------------------------------------------------------------
    # Gender Debias
    hd = HardDebias(criterion_name="gender",)
    hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    )

    gender_debiased_w2v = hd.transform(model, ignore=gender_specific, copy=True)

    assert model.name == "word2vec"
    assert gender_debiased_w2v.name == "word2vec_gender_debiased"

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(query_2, model, normalize=True)
    debiased_results = weat.run_query(query_2, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    # -----------------------------------------------------------------
    # Test target param
    hd = HardDebias(verbose=True, criterion_name="gender",)

    attributes = weat_wordset["pleasant_5"] + weat_wordset["unpleasant_5"]

    gender_debiased_w2v = hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    ).transform(model, target=attributes, copy=True)

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(query_2, model, normalize=True)
    debiased_results = weat.run_query(query_2, gender_debiased_w2v, normalize=True)
    assert debiased_results["weat"] - biased_results["weat"] < 0.000000

    # -----------------------------------------------------------------
    # Test ignore param
    hd = HardDebias(verbose=True, criterion_name="gender",)

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.
    targets = weat_wordset["male_names"] + weat_wordset["female_names"]
    attributes = weat_wordset["pleasant_5"] + weat_wordset["unpleasant_5"]
    gender_debiased_w2v = hd.fit(
        model, definitional_pairs, equalize_pairs=equalize_pairs,
    ).transform(model, ignore=gender_specific + targets + attributes, copy=True)

    biased_results = weat.run_query(query_1, model, normalize=True)
    debiased_results = weat.run_query(query_1, gender_debiased_w2v, normalize=True)

    assert debiased_results["weat"] - biased_results["weat"] < 0.000000

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

    assert model.name == "word2vec"
    assert gender_debiased_w2v.name == "word2vec_debiased"

    # -----------------------------------------------------------------
    # Test inplace (copy = False)
    hd = HardDebias(criterion_name="gender",)
    hd.fit(
        model, definitional_pairs=definitional_pairs, equalize_pairs=equalize_pairs,
    )

    gender_debiased_w2v = hd.transform(model, ignore=gender_specific, copy=False)
    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name


def test_multiclass_hard_debias_class(model):

    multiclass_debias_wordsets = fetch_debias_multiclass()
    weat_wordsets = load_weat()
    weat = WEAT()

    # -------------
    # Gender Debias
    gender_definitional_sets = multiclass_debias_wordsets["gender_definitional_sets"]
    gender_equalize_sets = list(
        multiclass_debias_wordsets["gender_analogy_templates"].values()
    )
    debiaswe_wordsets = fetch_debiaswe()
    gender_specific = debiaswe_wordsets["gender_specific"]

    mhd = MulticlassHardDebias(criterion_name="gender",)
    mhd.fit(
        model=model,
        definitional_sets=gender_definitional_sets,
        equalize_sets=gender_equalize_sets,
    )

    gender_debiased_w2v = mhd.transform(model, copy=True)

    assert model.name == "word2vec"
    assert gender_debiased_w2v.name == "word2vec_gender_debiased"

    gender_query_1 = Query(
        [weat_wordsets["male_names"], weat_wordsets["female_names"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )

    gender_query_2 = Query(
        [weat_wordsets["male_names"], weat_wordsets["female_names"]],
        [weat_wordsets["career"], weat_wordsets["family"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )

    # tests with WEAT
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

    gender_eval = np.array(multiclass_debias_wordsets["gender_eval_target"]).reshape(
        2, -1
    )
    gender_analogy_templates = np.array(
        list(multiclass_debias_wordsets["gender_analogy_templates"].values())
    )
    gender_query_MAC = Query(gender_eval, gender_analogy_templates)

    biased_results_2 = MAC().run_query(gender_query_MAC, model)
    debiased_results_2 = MAC().run_query(gender_query_MAC, gender_debiased_w2v)

    # in this case, closer to one is better:
    assert debiased_results_2["mac"] > biased_results_2["mac"]

    # ----------------
    # Ethnicity Debias

    ethnicity_definitional_sets = multiclass_debias_wordsets[
        "ethnicity_definitional_sets"
    ]
    ethnicity_equalize_sets = list(
        multiclass_debias_wordsets["ethnicity_analogy_templates"].values()
    )

    mhd = MulticlassHardDebias(verbose=True, criterion_name="ethnicity",)
    mhd.fit(
        model=model,
        definitional_sets=ethnicity_definitional_sets,
        equalize_sets=ethnicity_equalize_sets,
    )

    ethnicity_debiased_w2v = mhd.transform(model, copy=True)
    assert model.name == "word2vec"
    assert ethnicity_debiased_w2v.name == "word2vec_ethnicity_debiased"

    # test with weat

    ethnicity_query = Query(
        [
            weat_wordsets["european_american_names_5"],
            weat_wordsets["african_american_names_5"],
        ],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["european_american_names_5", "african_american_names_5"],
        ["pleasant_5", "unpleasant_5"],
    )

    biased_results = weat.run_query(ethnicity_query, model, normalize=True)
    debiased_results = weat.run_query(
        ethnicity_query, ethnicity_debiased_w2v, normalize=True
    )

    assert debiased_results["weat"] < biased_results["weat"]

    # -----------------------------------------------------------------
    # Test target param

    mhd = MulticlassHardDebias()

    attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]

    gender_debiased_w2v = mhd.fit(
        model,
        definitional_sets=gender_definitional_sets,
        equalize_sets=gender_equalize_sets,
    ).transform(model, target=attributes, copy=True)

    assert model.name == "word2vec"
    assert gender_debiased_w2v.name == "word2vec_debiased"

    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] < biased_results["weat"]

    biased_results = weat.run_query(gender_query_2, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_2, gender_debiased_w2v, normalize=True
    )
    assert debiased_results["weat"] - biased_results["weat"] < 0.000000

    # -----------------------------------------------------------------
    # Test ignore param
    mhd = MulticlassHardDebias(criterion_name="gender",)

    # in this test, the targets and attributes are included in the ignore list.
    # this implies that neither of these words should be subjected to debias and
    # therefore, both queries when executed with weat should return the same score.
    targets = weat_wordsets["male_names"] + weat_wordsets["female_names"]
    attributes = weat_wordsets["pleasant_5"] + weat_wordsets["unpleasant_5"]
    gender_debiased_w2v = mhd.fit(
        model,
        definitional_sets=gender_definitional_sets,
        equalize_sets=gender_equalize_sets,
    ).transform(model, ignore=gender_specific + targets + attributes, copy=True)

    biased_results = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results = weat.run_query(
        gender_query_1, gender_debiased_w2v, normalize=True
    )

    assert debiased_results["weat"] - biased_results["weat"] < 0.000000

    # -----------------------------------------------------------------

    # Test inplace (copy = False)
    mhd = MulticlassHardDebias(criterion_name="gender",)
    mhd.fit(
        model=model,
        definitional_sets=gender_definitional_sets,
        equalize_sets=gender_equalize_sets,
    )

    gender_debiased_w2v = mhd.transform(model, copy=False)

    assert model == gender_debiased_w2v
    assert model.wv == gender_debiased_w2v.wv
    assert model.name == gender_debiased_w2v.name
