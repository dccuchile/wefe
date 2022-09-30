"""Query testing module."""
from typing import Dict, List

import pytest
from wefe.datasets.datasets import load_weat
from wefe.query import Query


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


def test_create_query_input_checks():

    # target sets None
    with pytest.raises(TypeError, match="target_sets must be a*"):
        Query(None, None)

    # target sets int
    with pytest.raises(TypeError, match="target_sets must be a*"):
        Query(3, None)

    # target sets str
    with pytest.raises(TypeError, match="target_sets must be a*"):
        Query("1", None)

    # attribute sets None
    with pytest.raises(TypeError, match="attribute_sets must be a*"):
        Query([[""]], None)

    # attribute sets int
    with pytest.raises(TypeError, match="attribute_sets must be a*"):
        Query([[""]], 3)

    # attribute sets str
    with pytest.raises(TypeError, match="attribute_sets must be a*"):
        Query([[""]], "aaah")

    # target sets empty array
    with pytest.raises(Exception, match="target_sets must have*"):
        Query([], [])

    # target sets with wrong types
    with pytest.raises(TypeError, match="Each target set must be a*"):
        Query([None, ["a"]], [[""]])
    with pytest.raises(TypeError, match="Each target set must be a*"):
        Query([["a"], None], [[""]])

    with pytest.raises(TypeError, match="All elements in target set*"):
        Query([[None, "a"], ["b"]], [[""]])
    with pytest.raises(TypeError, match="All elements in target set*"):
        Query([["a"], ["b", 2]], [[""]])

    # attribute sets with wrong types
    with pytest.raises(TypeError, match="Each attribute set must be a*"):
        Query([["a"], ["b"]], [None, ["a"]])
    with pytest.raises(TypeError, match="Each attribute set must be a*"):
        Query([["a"], ["b"]], [["a"], None])

    with pytest.raises(TypeError, match="All elements in attribute set*"):
        Query([["a"], ["b"]], [[None, "a"]])
    with pytest.raises(TypeError, match="All elements in attribute set*"):
        Query([["a"], ["b"]], [["a"], ["b", 2]])


def test_create_query():

    weat_wordsets = load_weat()

    flowers = weat_wordsets["flowers"]
    insects = weat_wordsets["insects"]
    pleasant = weat_wordsets["pleasant_5"]
    unpleasant = weat_wordsets["unpleasant_5"]

    # create a query using the word sets:
    query = Query(
        [flowers, insects],
        [pleasant, unpleasant],
        ["Flowers", "Insects"],
        ["Pleasant", "Unpleasant"],
    )

    assert query.target_sets[0] == flowers
    assert query.target_sets[1] == insects
    assert query.attribute_sets[0] == pleasant
    assert query.attribute_sets[1] == unpleasant

    assert query.template[0] == 2
    assert query.template[1] == 2

    assert query.target_sets_names == ["Flowers", "Insects"]
    assert query.attribute_sets_names == ["Pleasant", "Unpleasant"]


def test_eq():

    weat = load_weat()

    flowers = weat["flowers"]
    insects = weat["insects"]
    weapons = weat["weapons"]
    pleasant_1 = weat["pleasant_5"]
    pleasant_2 = weat["pleasant_9"]
    unpleasant_1 = weat["unpleasant_5"]
    unpleasant_2 = weat["unpleasant_9"]

    query = Query([flowers, insects], [pleasant_1, unpleasant_1])
    query_2 = Query([flowers, weapons], [pleasant_1, unpleasant_1])
    query_3 = Query([weapons, flowers], [pleasant_1, unpleasant_1])
    query_4 = Query([flowers, insects], [pleasant_2, unpleasant_1])
    query_5 = Query([flowers, insects], [pleasant_1, unpleasant_2])

    assert query == query
    assert query_2 == query_2
    assert query_3 == query_3
    assert query_4 == query_4
    assert query_5 == query_5

    # type assertion
    assert query is not None
    assert query != []
    assert query != "123"
    assert query != 123
    assert query != {}

    assert query != query_2
    assert query != query_3
    assert query != query_4
    assert query != query_5
    assert query_2 != query_3
    assert query_2 != query_4
    assert query_2 != query_5
    assert query_3 != query_4
    assert query_3 != query_5
    assert query_4 != query_5

    # cardinality
    big_query_1 = Query([flowers, insects, weapons], [pleasant_1, unpleasant_1])
    big_query_2 = Query([flowers, insects], [pleasant_1, unpleasant_1, unpleasant_2])

    query != big_query_1
    query != big_query_2

    # names
    query_bad_name_1 = Query(
        [flowers, insects],
        [pleasant_1, unpleasant_1],
        ["Flawer", "Insects"],
        ["Pleasant 1", "Unpleasant 2"],
    )
    query_bad_name_2 = Query(
        [flowers, insects],
        [pleasant_1, unpleasant_1],
        ["Flowers", "Insec"],
        ["Pleasant 1", "Unpleasant 2"],
    )
    query_bad_name_3 = Query(
        [flowers, insects],
        [pleasant_1, unpleasant_1],
        ["Flowers", "Insects"],
        ["Pleas", "Unpleasant 2"],
    )
    query_bad_name_4 = Query(
        [flowers, insects],
        [pleasant_1, unpleasant_1],
        attribute_sets_names=["Pleasant 1", "asant 2"],
    )

    assert query_bad_name_1 != query
    assert query_bad_name_2 != query
    assert query_bad_name_3 != query
    assert query_bad_name_4 != query


def test_templates(weat_wordsets: Dict[str, List[str]]):

    flowers = weat_wordsets["flowers"]
    insects = weat_wordsets["insects"]
    weapons = weat_wordsets["weapons"]
    instruments = weat_wordsets["instruments"]
    pleasant = weat_wordsets["pleasant_5"]
    unpleasant = weat_wordsets["unpleasant_9"]

    query = Query(
        [flowers, insects, weapons, instruments],
        [pleasant, unpleasant],
        ["Flowers", "Insects", "Weapons", "Instruments"],
        ["Pleasant", "Unpleasant"],
    )

    # input validation
    with pytest.raises(
        TypeError, match="The new target cardinality (new_template[0])*"
    ):
        query.get_subqueries(("2", 2))
    with pytest.raises(
        TypeError, match="The new target cardinality (new_template[0])*"
    ):
        query.get_subqueries((None, 2))
    with pytest.raises(
        TypeError, match="The new attribute cardinality (new_template[1])*"
    ):
        query.get_subqueries((2, "2"))
    with pytest.raises(
        TypeError, match="The new attribute cardinality (new_template[1])*"
    ):
        query.get_subqueries((2, None))

    with pytest.raises(Exception, match="The new target cardinality*"):
        query.get_subqueries((5, 2))
    with pytest.raises(Exception, match="The new attribute cardinality*"):
        query.get_subqueries((4, 3))

    # equal subqueries
    assert query.get_subqueries((4, 2)) == [query]

    # target subqueries
    subqueries = query.get_subqueries((2, 2))
    assert len(subqueries) == 6
    target_names = [
        ["Flowers", "Insects"],
        ["Flowers", "Weapons"],
        ["Flowers", "Instruments"],
        ["Insects", "Weapons"],
        ["Insects", "Instruments"],
        ["Weapons", "Instruments"],
    ]

    for target_name, subquery in zip(target_names, subqueries):
        assert target_name == subquery.target_sets_names

    # attribute subqueries
    subqueries = query.get_subqueries((4, 1))
    attribute_names = [["Pleasant"], ["Unpleasant"]]
    assert len(subqueries) == 2

    for attribute_name, subquery in zip(attribute_names, subqueries):
        assert attribute_name == subquery.attribute_sets_names


def test_generate_query_name():

    weat_word_set = load_weat()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )

    assert query.query_name == "Flowers and Insects wrt Pleasant"

    query = Query(
        [weat_word_set["flowers"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers"],
        ["Pleasant"],
    )

    assert query.query_name == "Flowers wrt Pleasant"

    query = Query(
        [weat_word_set["flowers"], weat_word_set["instruments"]],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Flowers", "Instruments"],
        ["Pleasant", "Unpleasant"],
    )

    assert query.query_name == "Flowers and Instruments wrt Pleasant and " "Unpleasant"

    query = Query(
        [
            weat_word_set["flowers"],
            weat_word_set["instruments"],
            weat_word_set["weapons"],
            weat_word_set["insects"],
        ],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
        ["Flowers", "Instruments", "Weapons", "Insects"],
        ["Pleasant", "Unpleasant"],
    )

    assert (
        query.query_name == "Flowers, Instruments, Weapons and Insects "
        "wrt Pleasant and Unpleasant"
    )

    query = Query(
        [
            weat_word_set["flowers"],
            weat_word_set["instruments"],
            weat_word_set["weapons"],
            weat_word_set["insects"],
        ],
        [weat_word_set["pleasant_5"], weat_word_set["unpleasant_5"]],
    )

    assert (
        query.query_name == "Target set 0, Target set 1, Target set 2 "
        "and Target set 3 wrt Attribute set 0 and "
        "Attribute set 1"
    )

    query = Query(
        [
            weat_word_set["flowers"],
            weat_word_set["instruments"],
            weat_word_set["weapons"],
            weat_word_set["insects"],
        ],
        [],
    )

    assert (
        query.query_name == "Target set 0, Target set 1, Target set 2 "
        "and Target set 3"
    )


def test_wrong_target_and_attribute_sets_and_names(caplog):
    weat_word_set = load_weat()

    with pytest.raises(
        ValueError,
        match=(
            r"target_sets \(len=1\) does not have the same number of "
            r"elements as target_sets_names \(len=2\)"
        ),
    ):
        Query(
            [weat_word_set["flowers"]],
            [weat_word_set["pleasant_5"]],
            ["Flowers", "asdf"],
            ["Pleasant"],
        )
    with pytest.raises(
        ValueError,
        match=(
            r"attribute_sets \(len=1\) does not have the same number of elements as "
            r"attribute_sets_names \(len=2\)"
        ),
    ):
        Query(
            [weat_word_set["flowers"]],
            [weat_word_set["pleasant_5"]],
            ["Flowers"],
            ["Pleasant", "asdf"],
        )


def test_query__repr__():

    weat_word_set = load_weat()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )

    assert query.__repr__() == (
        "<Query: Flowers and Insects wrt Pleasant\n- Target sets: [['aster', 'clover'"
        ", 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid',"
        " 'rose', 'bluebell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup',"
        " 'daisy', 'lily', 'peony', 'violet', 'carnation', 'gladiola', 'magnolia',"
        " 'petunia', 'zinnia'], ['ant', 'caterpillar', 'flea', 'locust', 'spider',"
        " 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula', 'bee', 'cockroach',"
        " 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth',"
        " 'wasp', 'blackfly', 'dragonfly', 'horsefly', 'roach', 'weevil']]\n-"
        " Attribute sets:[['caress', 'freedom', 'health', 'love', 'peace', 'cheer',"
        " 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest',"
        " 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise',"
        " 'family', 'happy', 'laughter', 'paradise', 'vacation']]>"
    )

    query_2 = Query([[]], [])

    assert (
        query_2.__repr__()
        == "<Query: Target set 0\n- Target sets: [[]]\n- Attribute sets:[]>"
    )

    del query.target_sets
    assert query.__repr__() == "<Query with wrong __repr__>"


def test_query_dict():

    weat_word_set = load_weat()
    query = Query(
        [weat_word_set["flowers"], weat_word_set["insects"]],
        [weat_word_set["pleasant_5"]],
        ["Flowers", "Insects"],
        ["Pleasant"],
    )

    assert query.dict() == {
        "target_sets": [
            [
                "aster",
                "clover",
                "hyacinth",
                "marigold",
                "poppy",
                "azalea",
                "crocus",
                "iris",
                "orchid",
                "rose",
                "bluebell",
                "daffodil",
                "lilac",
                "pansy",
                "tulip",
                "buttercup",
                "daisy",
                "lily",
                "peony",
                "violet",
                "carnation",
                "gladiola",
                "magnolia",
                "petunia",
                "zinnia",
            ],
            [
                "ant",
                "caterpillar",
                "flea",
                "locust",
                "spider",
                "bedbug",
                "centipede",
                "fly",
                "maggot",
                "tarantula",
                "bee",
                "cockroach",
                "gnat",
                "mosquito",
                "termite",
                "beetle",
                "cricket",
                "hornet",
                "moth",
                "wasp",
                "blackfly",
                "dragonfly",
                "horsefly",
                "roach",
                "weevil",
            ],
        ],
        "attribute_sets": [
            [
                "caress",
                "freedom",
                "health",
                "love",
                "peace",
                "cheer",
                "friend",
                "heaven",
                "loyal",
                "pleasure",
                "diamond",
                "gentle",
                "honest",
                "lucky",
                "rainbow",
                "diploma",
                "gift",
                "honor",
                "miracle",
                "sunrise",
                "family",
                "happy",
                "laughter",
                "paradise",
                "vacation",
            ]
        ],
        "target_sets_names": ["Flowers", "Insects"],
        "attribute_sets_names": ["Pleasant"],
        "query_name": "Flowers and Insects wrt Pleasant",
        "template": (2, 1),
    }

    query_2 = Query([[]], [])

    assert query_2.dict() == {
        "target_sets": [[]],
        "attribute_sets": [],
        "target_sets_names": ["Target set 0"],
        "attribute_sets_names": [],
        "query_name": "Target set 0",
        "template": (1, 0),
    }
