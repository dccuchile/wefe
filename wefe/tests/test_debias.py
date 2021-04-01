import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from wefe.datasets import fetch_debiaswe
from wefe.debias.hard_debias import HardDebias
from wefe.word_embedding_model import WordEmbeddingModel


def test_hard_debias_class():

    # TODO: Cambiar por un comprimido...
    w2v_sm = KeyedVectors.load_word2vec_format(
        "./wefe/datasets/data/w2v_sm.bin", binary=True
    )
    we = WordEmbeddingModel(w2v_sm, "w2v_sm")

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

    definning_pairs_embeddings = hd._get_embeddings_from_pairs_sets(
        we, definitional_pairs, preprocessor_args, None, True, "definitional"
    )

    for pair, embedding_pair in zip(definitional_pairs, definning_pairs_embeddings):
        assert pair[0] in embedding_pair
        assert pair[1] in embedding_pair
        isinstance(embedding_pair[pair[0]], np.ndarray)
        isinstance(embedding_pair[pair[1]], np.ndarray)
        assert embedding_pair[pair[0]].shape == (300,)
        assert embedding_pair[pair[1]].shape == (300,)

    equalize_pairs_embeddings = hd._get_embeddings_from_pairs_sets(
        we, equalize_pairs, preprocessor_args, None, True, "equalize"
    )

    # TODO: FIx este test

    # for pair, embedding_pair in zip(equalize_pairs, equalize_pairs_embeddings):
    #     assert pair[0] in embedding_pair
    #     assert pair[1] in embedding_pair
    #     isinstance(embedding_pair[pair[0]], np.ndarray)
    #     isinstance(embedding_pair[pair[1]], np.ndarray)
    #     assert embedding_pair[pair[0]].shape == (300,)
    #     assert embedding_pair[pair[1]].shape == (300,)

    hd.run_debias(we, definitional_pairs, gender_specific, equalize_pairs)
