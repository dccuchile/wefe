"""Test configurations and fixtures."""
from typing import Dict, List

import pytest
from wefe.datasets.datasets import fetch_debiaswe, load_weat
from wefe.query import Query
from wefe.utils import load_test_model
from wefe.word_embedding_model import WordEmbeddingModel


@pytest.fixture
def model() -> WordEmbeddingModel:
    """Load a subset of Word2vec as a testing model.

    Returns
    -------
    WordEmbeddingModel
        The loaded testing model.
    """
    return load_test_model()


@pytest.fixture
def weat_wordsets() -> Dict[str, List[str]]:
    """Load the word sets used in WEAT original work.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary that map a word set name to a set of words.
    """
    weat_wordsets = load_weat()
    return weat_wordsets


@pytest.fixture
def gender_specific():
    debiaswe_wordsets = fetch_debiaswe()
    gender_specific = debiaswe_wordsets["gender_specific"]
    return gender_specific


@pytest.fixture
def query_2t1a_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    weat_wordsets = load_weat()

    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )
    return query


@pytest.fixture
def query_2t2a_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    """Generate a Flower and Insects wrt Pleasant vs Unpleasant test query.

    Parameters
    ----------
    weat_wordsets : Dict[str, List[str]]
        The word sets used in WEAT original work.

    Returns
    -------
    Query
        The generated query.
    """
    query = Query(
        [weat_wordsets["flowers"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )
    return query


@pytest.fixture
def query_3t2a_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    query = Query(
        [
            weat_wordsets["flowers"],
            weat_wordsets["insects"],
            weat_wordsets["instruments"],
        ],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Weapons", "Instruments"],
        ["Pleasant", "Unpleasant"],
    )

    return query


@pytest.fixture
def query_4t2a_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    query = Query(
        [
            weat_wordsets["flowers"],
            weat_wordsets["insects"],
            weat_wordsets["instruments"],
            weat_wordsets["weapons"],
        ],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects", "Instruments", "Weapons"],
        ["Pleasant", "Unpleasant"],
    )

    return query


@pytest.fixture
def query_1t4_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    query = Query(
        [weat_wordsets["flowers"]],
        [
            weat_wordsets["pleasant_5"],
            weat_wordsets["pleasant_9"],
            weat_wordsets["unpleasant_5"],
            weat_wordsets["unpleasant_9"],
        ],
        ["Flowers"],
        ["Pleasant 5 ", "Pleasant 9", "Unpleasant 5", "Unpleasant 9"],
    )
    return query


@pytest.fixture
def query_2t1a_lost_vocab_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    query = Query(
        [["bla", "asd"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )

    return query


@pytest.fixture
def query_2t2a_lost_vocab_1(weat_wordsets: Dict[str, List[str]]) -> Query:
    query = Query(
        [["bla", "asd"], weat_wordsets["insects"]],
        [weat_wordsets["pleasant_5"], weat_wordsets["unpleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )

    return query
