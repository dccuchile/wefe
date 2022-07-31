"""Base debias testing."""
from typing import List

import pytest
from wefe.debias.base_debias import BaseDebias
from wefe.word_embedding_model import WordEmbeddingModel


def test_base_debias():

    bd = BaseDebias()
    with pytest.raises(NotImplementedError):
        bd.fit(None)
    with pytest.raises(NotImplementedError):
        bd.transform(None)
    with pytest.raises(NotImplementedError):
        bd.fit_transform(None)


def test_check_transform_args_wrong_inputs(
    model: WordEmbeddingModel, gender_specific: List[str]
):

    bd = BaseDebias()

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


def test_check_transform_args_ok_inputs(
    model: WordEmbeddingModel, gender_specific: List[str]
):

    bd = BaseDebias()
    bd._check_transform_args(
        model=model, target=gender_specific, ignore=["some", "words"], copy=True
    )

    bd._check_transform_args(
        model=model, target=gender_specific, ignore=None, copy=True
    )

    bd._check_transform_args(
        model=model, target=gender_specific, ignore=None, copy=False
    )

    bd._check_transform_args(model=model, target=None, ignore=None, copy=False)
