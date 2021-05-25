import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors

from wefe.datasets import fetch_debiaswe, load_weat
from wefe.debias.hard_debias import HardDebias
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics import WEAT
from wefe.query import Query


@pytest.fixture
def model():
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
        debias_criterion_name="gender",
        verbose=True,
    )
    debiased_w2v = hd.transform(model, ignore=gender_specific, copy=True, verbose=True,)

    weat_word_set = load_weat()
    weat = WEAT()
    query = Query(
        [weat_word_set["male_names"], weat_word_set["female_names"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )
    biased_results = weat.run_query(query, model)
    debiased_results = weat.run_query(query, debiased_w2v)

    assert debiased_results["weat"] < biased_results["weat"]
