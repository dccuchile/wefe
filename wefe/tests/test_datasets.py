import pytest

from ..datasets.datasets import fetch_bingliu, fetch_debiaswe, fetch_eds, fetch_debias_multiclass, load_weat


def test_fetch_bingliu():
    bingliu = fetch_bingliu()
    assert isinstance(bingliu, dict)
    assert list(bingliu.keys()) == ['Positive words', 'Negative words']
    assert len(list(bingliu.keys())) == 2

    for key in bingliu:
        assert len(bingliu[key]) > 0


def test_fetch_eds():
    eds = fetch_eds()
    assert isinstance(eds, dict)
    assert list(eds.keys()) == [
        'Adjectives appearance', 'Adjectives otherization', 'Adjectives sensitive', 'Names asian', 'Names black',
        'Names chinese', 'Names hispanic', 'Names russian', 'Names white', 'Words christianity', 'Words islam',
        'Words terrorism', 'Male Occupations', 'Female Occupations', 'Male terms', 'Female terms',
        'Adjectives intelligence'
    ]
    assert len(list(eds.keys())) == 17

    for key in eds:
        assert len(eds[key]) > 0


def test_fetch_debiaswe():
    debiaswe = fetch_debiaswe()
    assert isinstance(debiaswe, dict)
    assert list(debiaswe.keys()) == ['Male terms', 'Female terms', 'Male related words', 'Female related words']
    assert len(list(debiaswe.keys())) == 4

    for key in debiaswe:
        assert len(debiaswe[key]) > 0


def test_fetch_debias_multiclass():
    debias_multiclass = fetch_debias_multiclass()
    assert isinstance(debias_multiclass, dict)
    assert list(debias_multiclass.keys()) == [
        'Male terms', 'Female terms', 'Male related words', 'Female related words', 'Black terms', 'White terms',
        'Asian terms', 'Black related words', 'White related words', 'Asian related words', 'Judaism terms',
        'Christianity terms', 'Islam terms', 'Jew related words', 'Christian related words', 'Muslim related words'
    ]
    assert len(list(debias_multiclass.keys())) == 16

    for key in debias_multiclass:
        assert len(debias_multiclass[key]) > 0


def test_load_weat():
    weat = load_weat()
    assert isinstance(weat, dict)
    assert list(weat.keys()) == [
        'Flowers', 'Insects', 'Pleasant 5', 'Unpleasant 5', 'Instruments', 'Weapons', 'European american names 5',
        'African american names 5', 'European american names 7', 'African american names 7', 'Pleasant 9',
        'Unpleasant 9', 'Male names', 'Female names', 'Career', 'Family', 'Math', 'Arts', 'Male terms', 'Female terms',
        'Science', 'Arts 2', 'Male terms 2', 'Female terms 2', 'Mental disease', 'Physical disease', 'Temporary',
        'Permanent', 'Young peoples names', 'Old peoples names'
    ]
    for key in weat:
        assert len(weat[key]) > 0