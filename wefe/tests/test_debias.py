import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors

from wefe.datasets import fetch_debiaswe
from wefe.debias.hard_debias import HardDebias
from wefe.word_embedding_model import WordEmbeddingModel


@pytest.fixture
def word2vec_sm():
    w2v = KeyedVectors.load_word2vec_format("./wefe/tests/w2v_sm.bin", binary=True)
    return WordEmbeddingModel(w2v, "word2vec")


def test_hard_debias_class(word2vec_sm):

    preprocessor_args = {
        "strip_accents": False,
        "lowercase": False,
        "preprocessor": None,
    }

    debiaswe_wordsets = fetch_debiaswe()

    definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    equalize_pairs = debiaswe_wordsets["equalize_pairs"]
    gender_specific = debiaswe_wordsets["gender_specific"]

    hd = HardDebias()

    hd.run_debias(
        word2vec_sm, definitional_pairs, gender_specific, equalize_pairs, inplace=False
    )

