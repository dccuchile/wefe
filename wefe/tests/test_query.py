import pytest
from ..datasets.datasets import load_weat
from ..query import Query


def test_create_query_input_verifications():

    # target sets None
    with pytest.raises(TypeError, match='target_sets must be a*'):
        Query(None, None)

    # target sets int
    with pytest.raises(TypeError, match='target_sets must be a*'):
        Query(3, None)

    # target sets str
    with pytest.raises(TypeError, match='target_sets must be a*'):
        Query('1', None)

    # attribute sets None
    with pytest.raises(TypeError, match='attribute_sets must be a*'):
        Query([['']], None)

    # attribute sets int
    with pytest.raises(TypeError, match='attribute_sets must be a*'):
        Query([['']], 3)

    # attribute sets str
    with pytest.raises(TypeError, match='attribute_sets must be a*'):
        Query([['']], 'aaah')

    # target sets empty array
    with pytest.raises(Exception, match='target_sets must have*'):
        Query([], [])

    # target sets with wrong types
    with pytest.raises(TypeError, match='Each target set must be a*'):
        Query([None, ['a']], [['']])
    with pytest.raises(TypeError, match='Each target set must be a*'):
        Query([['a'], None], [['']])

    with pytest.raises(TypeError, match='All elements in target set*'):
        Query([[None, 'a'], ['b']], [['']])
    with pytest.raises(TypeError, match='All elements in target set*'):
        Query([['a'], ['b', 2]], [['']])

    # attribute sets with wrong types
    with pytest.raises(TypeError, match='Each attribute set must be a*'):
        Query([['a'], ['b']], [None, ['a']])
    with pytest.raises(TypeError, match='Each attribute set must be a*'):
        Query([['a'], ['b']], [['a'], None])

    with pytest.raises(TypeError, match='All elements in attribute set*'):
        Query([['a'], ['b']], [[None, 'a']])
    with pytest.raises(TypeError, match='All elements in attribute set*'):
        Query([['a'], ['b']], [['a'], ['b', 2]])


def test_create_query():

    # create a real query:
    weat = load_weat()

    flowers = weat['Flowers']
    insects = weat['Insects']
    pleasant = weat['Pleasant 5']
    unpleasant = weat['Unpleasant 5']

    query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Insects'], ['Pleasant', 'Unpleasant'])

    assert query.target_sets_[0] == flowers
    assert query.target_sets_[1] == insects
    assert query.attribute_sets_[0] == pleasant
    assert query.attribute_sets_[1] == unpleasant

    assert query.template_[0] == 2
    assert query.template_[1] == 2

    assert query.target_sets_names_ == ['Flowers', 'Insects']
    assert query.attribute_sets_names_ == ['Pleasant', 'Unpleasant']


def test_eq():

    weat = load_weat()

    flowers = weat['Flowers']
    insects = weat['Insects']
    weapons = weat['Weapons']
    pleasant_1 = weat['Pleasant 5']
    pleasant_2 = weat['Pleasant 9']
    unpleasant_1 = weat['Unpleasant 5']
    unpleasant_2 = weat['Unpleasant 9']

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
    assert query != []
    assert query != None
    assert query != '123'
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
    query_bad_name_1 = Query([flowers, insects], [pleasant_1, unpleasant_1], ['Flawer', 'Insects'],
                             ['Pleasant 1', 'Unpleasant 2'])
    query_bad_name_2 = Query([flowers, insects], [pleasant_1, unpleasant_1], ['Flowers', 'Insec'],
                             ['Pleasant 1', 'Unpleasant 2'])
    query_bad_name_3 = Query([flowers, insects], [pleasant_1, unpleasant_1], ['Flowers', 'Insects'],
                             ['Pleas', 'Unpleasant 2'])
    query_bad_name_4 = Query([flowers, insects], [pleasant_1, unpleasant_1], ['Flowers', 'Insects'],
                             ['Pleasant 1', 'asant 2'])

    assert query_bad_name_1 != query
    assert query_bad_name_2 != query
    assert query_bad_name_3 != query
    assert query_bad_name_4 != query


def test_templates():

    weat = load_weat()

    flowers = weat['Flowers']
    insects = weat['Insects']
    weapons = weat['Weapons']
    instruments = weat['Instruments']
    pleasant = weat['Pleasant 5']
    unpleasant = weat['Unpleasant 9']

    query = Query([flowers, insects, weapons, instruments], [pleasant, unpleasant],
                  ['Flowers', 'Insects', 'Weapons', 'Instruments'], ['Pleasant', 'Unpleasant'])

    # input validation
    with pytest.raises(TypeError, match='The new target cardinality (new_template[0])*'):
        query.generate_subqueries(('2', 2))
    with pytest.raises(TypeError, match='The new target cardinality (new_template[0])*'):
        query.generate_subqueries((None, 2))
    with pytest.raises(TypeError, match='The new attribute cardinality (new_template[1])*'):
        query.generate_subqueries((2, '2'))
    with pytest.raises(TypeError, match='The new attribute cardinality (new_template[1])*'):
        query.generate_subqueries((2, None))

    with pytest.raises(Exception, match='The new target cardinality*'):
        query.generate_subqueries((5, 2))
    with pytest.raises(Exception, match='The new attribute cardinality*'):
        query.generate_subqueries((4, 3))

    # equal subqueries
    assert query.generate_subqueries((4, 2)) == [query]

    # target subqueries
    subqueries = query.generate_subqueries((2, 2))
    assert len(subqueries) == 6
    target_names = [['Flowers', 'Insects'], ['Flowers', 'Weapons'], ['Flowers', 'Instruments'], ['Insects', 'Weapons'],
                    ['Insects', 'Instruments'], ['Weapons', 'Instruments']]

    for target_name, subquery in zip(target_names, subqueries):
        assert target_name == subquery.target_sets_names_

    # attribute subqueries
    subqueries = query.generate_subqueries((4, 1))
    attribute_names = [['Pleasant'], ['Unpleasant']]
    assert len(subqueries) == 2

    for attribute_name, subquery in zip(attribute_names, subqueries):
        assert attribute_name == subquery.attribute_sets_names_


def test_generate_query_name():

    weat_word_set = load_weat()
    query = Query([weat_word_set['Flowers'], weat_word_set['Insects']], [weat_word_set['Pleasant 5']],
                  ['Flowers', 'Insects'], ['Pleasant'])

    assert query.query_name_ == 'Flowers and Insects wrt Pleasant'

    query = Query([weat_word_set['Flowers']], [weat_word_set['Pleasant 5']], ['Flowers'], ['Pleasant'])

    assert query.query_name_ == 'Flowers wrt Pleasant'

    query = Query([weat_word_set['Flowers'], weat_word_set['Instruments']],
                  [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Instruments'],
                  ['Pleasant', 'Unpleasant'])

    assert query.query_name_ == 'Flowers and Instruments wrt Pleasant and Unpleasant'

    query = Query(
        [weat_word_set['Flowers'], weat_word_set['Instruments'], weat_word_set['Weapons'], weat_word_set['Insects']],
        [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Instruments', 'Weapons', 'Insects'],
        ['Pleasant', 'Unpleasant'])

    assert query.query_name_ == 'Flowers, Instruments, Weapons and Insects wrt Pleasant and Unpleasant'

    query = Query(
        [weat_word_set['Flowers'], weat_word_set['Instruments'], weat_word_set['Weapons'], weat_word_set['Insects']],
        [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']])

    assert query.query_name_ == 'Target set 0, Target set 1, Target set 2 and Target set 3 wrt Attribute set 0 and Attribute set 1'