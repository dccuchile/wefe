"""Repulsion Attraction Neutralization tests set."""
from typing import Dict, List

import numpy as np

from wefe.debias.repulsion_attraction_neutralization import (
    RepulsionAttractionNeutralization,
)
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_hard_debias_target_param(
    model: WordEmbeddingModel,
    gender_query_1: Query,
    gender_query_2: Query,
    control_query_1: Query,
    weat_wordsets: Dict[str, List[str]],
    definitional_pairs: List[List[str]],
):
    weat = WEAT()

    attribute_words = weat_wordsets["career"] + weat_wordsets["family"]
    attribute_words.remove("family")
    attribute_words.remove("executive")

    ran = RepulsionAttractionNeutralization(
        criterion_name="gender",
    )
    ran.fit(model, definitional_pairs=definitional_pairs)

    gender_debiased_w2v = ran.transform(model, target=attribute_words, copy=True)

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


# def test_repulsion_attraction_repulsion_class(
#     model: WordEmbeddingModel,
#     gender_query_1: Query,
#     gender_query_2: Query,
#     definitional_pairs: List[List[str]],
#     gender_specific: List[str],
# ):

#     weat = WEAT()

#     # -----------------------------------------------------------------
#     # Gender Debias
#     ran = RepulsionAttractionNeutralization(criterion_name="gender",)
#     ran.fit(model, definitional_pairs=definitional_pairs)

#     gender_debiased_w2v = ran.transform(model, ignore=gender_specific, copy=True)

#     assert model.name == "test_w2v"
#     assert gender_debiased_w2v.name == "test_w2v_gender_debiased"

#     # check gender query 1 (Male Names and Female Names wrt Pleasant and Unpleasant)
#     # in original and debiased model.
#     biased_results = weat.run_query(gender_query_1, model, normalize=True)
#     debiased_results = weat.run_query(
#         gender_query_1, gender_debiased_w2v, normalize=True
#     )
#     assert abs(debiased_results["weat"]) < abs(biased_results["weat"])

#     # check gender query 2 (Male Names and Female Names wrt Career and Family)
#     # in original and debiased model.
#     biased_results = weat.run_query(gender_query_2, model, normalize=True)
#     debiased_results = weat.run_query(
#         gender_query_2, gender_debiased_w2v, normalize=True
#     )
#     assert abs(debiased_results["weat"]) < abs(biased_results["weat"])
