from ..datasets.datasets import load_bingliu, fetch_debiaswe, fetch_eds,\
                                fetch_debias_multiclass, load_weat


def test_load_bingliu():
    bingliu = load_bingliu()
    assert isinstance(bingliu, dict)
    assert list(bingliu.keys()) == ['positive_words', 'negative_words']
    assert len(list(bingliu.keys())) == 2

    for key in bingliu:
        assert len(bingliu[key]) > 0


def test_fetch_eds():
    eds = fetch_eds()
    assert isinstance(eds, dict)
    assert list(eds.keys()) == [
        'adjectives_appearance', 'adjectives_otherization',
        'adjectives_sensitive', 'names_asian', 'names_black', 'names_chinese',
        'names_hispanic', 'names_russian', 'names_white', 'words_christianity',
        'words_islam', 'words_terrorism', 'male_occupations',
        'female_occupations', 'occupations_white', 'occupations_black',
        'occupations_asian', 'occupations_hispanic', 'male_terms',
        'female_terms', 'adjectives_intelligence'
    ]
    assert len(list(eds.keys())) == 21

    for key in eds:
        assert len(eds[key]) > 0


def test_fetch_debiaswe():
    debiaswe = fetch_debiaswe()
    assert isinstance(debiaswe, dict)
    assert list(debiaswe.keys()) == [
        'male_terms', 'female_terms', 'male_related_words',
        'female_related_words'
    ]
    assert len(list(debiaswe.keys())) == 4

    for key in debiaswe:
        assert len(debiaswe[key]) > 0


def test_fetch_debias_multiclass():
    debias_multiclass = fetch_debias_multiclass()
    assert isinstance(debias_multiclass, dict)
    assert list(debias_multiclass.keys()) == [
        'male_terms', 'female_terms', 'male_roles', 'female_roles',
        'black_terms', 'white_terms', 'asian_terms', 'black_related_words',
        'white_related_words', 'asian_related_words', 'judaism_terms',
        'christianity_terms', 'islam_terms', 'greed', 'conservative',
        'terrorism'
    ]
    assert len(list(debias_multiclass.keys())) == 16

    for key in debias_multiclass:
        assert len(debias_multiclass[key]) > 0


def test_load_weat():
    weat = load_weat()
    assert isinstance(weat, dict)
    assert list(weat.keys()) == [
        'flowers', 'insects', 'pleasant_5', 'unpleasant_5', 'instruments',
        'weapons', 'european_american_names_5', 'african_american_names_5',
        'european_american_names_7', 'african_american_names_7', 'pleasant_9',
        'unpleasant_9', 'male_names', 'female_names', 'career', 'family',
        'math', 'arts', 'male_terms', 'female_terms', 'science', 'arts_2',
        'male_terms_2', 'female_terms_2', 'mental_disease', 'physical_disease',
        'temporary', 'permanent', 'young_people_names', 'old_people_names'
    ]
    for key in weat:
        assert len(weat[key]) > 0
