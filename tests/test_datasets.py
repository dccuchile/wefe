from wefe.datasets.datasets import (
    load_bingliu,
    fetch_debiaswe,
    fetch_eds,
    fetch_gn_glove,
    fetch_debias_multiclass,
    load_weat,
)


def test_load_bingliu():
    bingliu = load_bingliu()
    assert isinstance(bingliu, dict)
    assert list(bingliu.keys()) == ["positive_words", "negative_words"]
    assert len(list(bingliu.keys())) == 2

    for set_name, set_ in bingliu.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_fetch_eds():
    eds_dataset = fetch_eds()
    assert isinstance(eds_dataset, dict)
    assert list(eds_dataset.keys()) == [
        "adjectives_appearance",
        "adjectives_otherization",
        "adjectives_sensitive",
        "names_asian",
        "names_black",
        "names_chinese",
        "names_hispanic",
        "names_russian",
        "names_white",
        "words_christianity",
        "words_islam",
        "words_terrorism",
        "male_occupations",
        "female_occupations",
        "occupations_white",
        "occupations_black",
        "occupations_asian",
        "occupations_hispanic",
        "male_terms",
        "female_terms",
        "adjectives_intelligence",
    ]
    assert len(list(eds_dataset.keys())) == 21

    for set_name, set_ in eds_dataset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_fetch_debiaswe():
    debiaswe_datatset = fetch_debiaswe()
    assert isinstance(debiaswe_datatset, dict)
    assert list(debiaswe_datatset.keys()) == [
        "male_terms",
        "female_terms",
        "definitional_pairs",
        "equalize_pairs",
        "gender_specific",
        "professions",
    ]
    assert len(list(debiaswe_datatset.keys())) == 6

    for set_name, set_ in debiaswe_datatset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, (str, list))
            assert len(word) > 0


def test_fetch_debias_multiclass():
    debias_multiclass_dataset = fetch_debias_multiclass()
    assert isinstance(debias_multiclass_dataset, dict)
    assert list(debias_multiclass_dataset.keys()) == [
        "male_terms",
        "female_terms",
        "male_roles",
        "female_roles",
        "black_terms",
        "white_terms",
        "asian_terms",
        "black_biased_words",
        "white_biased_words",
        "asian_biased_words",
        "judaism_terms",
        "christianity_terms",
        "islam_terms",
        "greed",
        "conservative",
        "terrorism",
        "gender_definitional_sets",
        "ethnicity_definitional_sets",
        "religion_definitional_sets",
        "gender_analogy_templates",
        "ethnicity_analogy_templates",
        "religion_analogy_templates",
        "gender_eval_target",
        "ethnicity_eval_target",
        "religion_eval_target",
    ]

    assert len(list(debias_multiclass_dataset.keys())) == 25

    for set_name, set_ in debias_multiclass_dataset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, (list, dict))
        if isinstance(set_, list):
            assert len(set_) > 0
            for word in set_:
                assert isinstance(word, (str, list))
                assert len(word) > 0


def test_load_weat():
    weat = load_weat()
    assert isinstance(weat, dict)
    assert list(weat.keys()) == [
        "flowers",
        "insects",
        "pleasant_5",
        "unpleasant_5",
        "instruments",
        "weapons",
        "european_american_names_5",
        "african_american_names_5",
        "european_american_names_7",
        "african_american_names_7",
        "pleasant_9",
        "unpleasant_9",
        "male_names",
        "female_names",
        "career",
        "family",
        "math",
        "arts",
        "male_terms",
        "female_terms",
        "science",
        "arts_2",
        "male_terms_2",
        "female_terms_2",
        "mental_disease",
        "physical_disease",
        "temporary",
        "permanent",
        "young_people_names",
        "old_people_names",
    ]
    for set_name, set_ in weat.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_load_gn_glove():
    gn_glove_words = fetch_gn_glove()
    assert isinstance(gn_glove_words, dict)
    assert list(gn_glove_words.keys()) == ["male_terms", "female_terms"]
    for set_name, set_ in gn_glove_words.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0

