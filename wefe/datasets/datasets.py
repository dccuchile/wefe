"""Module with functions to load datasets and sets of words related to bias."""
import json
import urllib.request
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pkg_resources


def fetch_eds(
    occupations_year: int = 2015, top_n_race_occupations: int = 10
) -> Dict[str, List[str]]:
    """Fetch the word sets used in the experiments of the work *Word Embeddings
    *Quantify 100 Years Of Gender And Ethnic Stereotypes*.

    This dataset includes gender (male, female), ethnicity (asian, black, white) and
    religion (christianity and islam) and adjetives (appearence, intelligence,
    otherization, sensitive) word sets.

    References
    ----------
    Word Embeddings quantify 100 years of gender and ethnic stereotypes.
    Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Proceedings of
    the National Academy of Sciences, 115(16), E3635-E3644.


    Parameters
    ----------
    occupations_year : int, optional
        The year of the census for the occupations file.
        Available years: {'1850', '1860', '1870', '1880', '1900', '1910',
        '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990',
        '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
        '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'}
        , by default 2015
    top_n_race_occupations : int, optional
        The year of the census for the occupations file.
        The number of occupations by race, by default 10

    Returns
    -------
    dict
        A dictionary with the word sets.
    """

    EDS_BASE_URL = (
        "https://raw.githubusercontent.com/nikhgarg/"
        "EmbeddingDynamicStereotypes/master/data/"
    )
    EDS_WORD_SETS_NAMES = [
        "adjectives_appearance.txt",
        "adjectives_intelligencegeneral.txt",
        "adjectives_otherization.txt",
        "adjectives_sensitive.txt",
        "female_pairs.txt",
        "male_pairs.txt",
        "names_asian.txt",
        "names_black.txt",
        "names_chinese.txt",
        "names_hispanic.txt",
        "names_russian.txt",
        "names_white.txt",
        "words_christianity.txt",
        "words_islam.txt",
        "words_terrorism.txt",
    ]
    # read the word sets from the source.
    word_sets = []
    for EDS_words_set_name in EDS_WORD_SETS_NAMES:
        name = EDS_words_set_name.replace(".txt", "")
        word_sets.append(pd.read_csv(EDS_BASE_URL + EDS_words_set_name, names=[name]))

    word_sets_dict = pd.concat(word_sets, sort=False, axis=1).to_dict(orient="list")

    # turn the dataframe into a python dict without nan.
    for dataset_name in word_sets_dict:
        word_sets_dict[dataset_name] = list(
            filter(lambda x: not pd.isnull(x), word_sets_dict[dataset_name])
        )

    # ---- Occupations by Gender ----

    # fetch occupations by gender
    gender_occupations = pd.read_csv(
        EDS_BASE_URL + "occupation_percentages_gender_occ1950.csv"
    )
    # filter by year
    gender_occupations = gender_occupations[
        gender_occupations["Census year"] == occupations_year
    ]

    # get male occupations
    male_occupations = gender_occupations[
        gender_occupations["Male"] >= gender_occupations["Female"]
    ]
    male_occupations = male_occupations["Occupation"].values.tolist()

    # get female occupations
    female_occupations = gender_occupations[
        gender_occupations["Male"] < gender_occupations["Female"]
    ]
    female_occupations = female_occupations["Occupation"].values.tolist()

    word_sets_dict["male_occupations"] = male_occupations
    word_sets_dict["female_occupations"] = female_occupations

    # ---- Occupations by Ethnicity ----

    occupations = pd.read_csv(EDS_BASE_URL + "occupation_percentages_race_occ1950.csv")
    occupations_filtered = occupations[occupations["Census year"] == occupations_year]
    occupations_white = (
        occupations_filtered.sort_values("white")
        .head(top_n_race_occupations)[["Occupation"]]
        .values.T[0]
        .tolist()
    )
    occupations_black = (
        occupations_filtered.sort_values("black")
        .head(top_n_race_occupations)[["Occupation"]]
        .values.T[0]
        .tolist()
    )
    occupations_asian = (
        occupations_filtered.sort_values("asian")
        .head(top_n_race_occupations)[["Occupation"]]
        .values.T[0]
        .tolist()
    )
    occupations_hispanic = (
        occupations_filtered.sort_values("hispanic")
        .head(top_n_race_occupations)[["Occupation"]]
        .values.T[0]
        .tolist()
    )

    # add loaded sets to the dataset
    word_sets_dict["occupations_white"] = occupations_white
    word_sets_dict["occupations_black"] = occupations_black
    word_sets_dict["occupations_asian"] = occupations_asian
    word_sets_dict["occupations_hispanic"] = occupations_hispanic

    # rename some sets
    word_sets_dict["male_terms"] = word_sets_dict.pop("male_pairs")
    word_sets_dict["female_terms"] = word_sets_dict.pop("female_pairs")
    word_sets_dict["adjectives_intelligence"] = word_sets_dict.pop(
        "adjectives_intelligencegeneral"
    )

    return word_sets_dict


def fetch_debiaswe() -> Dict[str, Union[List[str], list]]:
    """Fetch the word sets used in the paper Man is to Computer Programmer as
    Woman is to Homemaker? from the source. It includes gender (male, female)
    terms and related word sets.

    References
    ----------

    Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings by Tolga Bolukbasi, Kai-Wei Chang, James Zou,
    Venkatesh Saligrama, and Adam Kalai.
    Proceedings of NIPS 2016.

    Returns
    -------
    dict
        A dictionary in which each key correspond to the name of the set and
        its values correspond to the word set.
    """

    DEBIAS_WE_BASE_URL = (
        "https://raw.githubusercontent.com/tolga-b/debiaswe/master/data/"
    )

    DEBIAS_WE_WORD_SETS = [
        "definitional_pairs.json",
        "equalize_pairs.json",
        "gender_specific_full.json",
        "professions.json",
    ]

    with urllib.request.urlopen(
        DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[0]
    ) as json_file:
        definitional_pairs = json.loads(json_file.read().decode())
        male_words = [p[0] for p in definitional_pairs]
        female_words = [p[1] for p in definitional_pairs]

    with urllib.request.urlopen(
        DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[1]
    ) as json_file:
        equalize_pairs = json.loads(json_file.read().decode())
    with urllib.request.urlopen(
        DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[2]
    ) as json_file:
        gender_specific = json.loads(json_file.read().decode())
    with urllib.request.urlopen(
        DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[3]
    ) as json_file:
        professions = json.loads(json_file.read().decode())

    word_sets_dict = {
        "male_terms": male_words,
        "female_terms": female_words,
        "definitional_pairs": definitional_pairs,
        "equalize_pairs": equalize_pairs,
        "gender_specific": gender_specific,
        "professions": professions,
    }

    return word_sets_dict


def load_bingliu() -> Dict[str, List[str]]:
    """Load the Bing-Liu sentiment lexicon.

    References
    ----------
    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
    Proceedings of the ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
    Washington, USA.

    Returns
    -------
    dict
        A dictionary with the positive and negative words.
    """
    # extract the file
    resource_package = __name__
    resource_neg_path = "/".join(("data", "negative-words.txt"))
    bingliu_neg_bytes = pkg_resources.resource_stream(
        resource_package, resource_neg_path
    )

    resource_pos_path = "/".join(("data", "positive-words.txt"))
    bingliu_pos_bytes = pkg_resources.resource_stream(
        resource_package, resource_pos_path
    )

    negative = pd.read_csv(
        bingliu_neg_bytes, sep="\n", header=None, names=["word"], encoding="latin-1"
    )
    negative_cleaned = negative.loc[30:].values.flatten().tolist()
    positive = pd.read_csv(
        bingliu_pos_bytes, sep="\n", header=None, names=["word"], encoding="latin-1"
    )
    positive_cleaned = positive.loc[29:].values.flatten().tolist()

    bingliu_lexicon = {
        "positive_words": positive_cleaned,
        "negative_words": negative_cleaned,
    }

    return bingliu_lexicon


def fetch_debias_multiclass() -> Dict[str, Union[List[str], list]]:
    """Fetch the word sets used in the paper Black Is To Criminals as Caucasian
    Is To Police: Detecting And Removing Multiclass Bias In Word Embeddings.

    This dataset contains gender (male, female), ethnicity (asian, black, white) and
    religion (christianity, judaism and islam) word sets.
    This helper allow accessing independently to each of the word sets (to be used
    as target or attribute sets in metrics) as well as to access them in the original
    format (to be used in debiasing methods).
    The dictionary keys whose names contain definitional sets and analogies
    templates are the keys that point to the original format focused on debiasing.
    
    References
    ----------
    | [1]: Thomas Manzini, Lim Yao Chong,Alan W Black, and Yulia Tsvetkov.
    | Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass
    | Bias in Word Embeddings.
    | In Proceedings of the 2019 Conference of the North American Chapter of the
    | Association for Computational Linguistics:
    | Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621,
    | Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
    | [2]: https://github.com/TManzini/DebiasMulticlassWordEmbedding/blob/master/Debiasing/evalBias.py


    Returns
    -------
    dict
        A dictionary in which each key correspond to the name of the set and
        its values correspond to the word set.

    """

    BASE_URL = (
        "https://raw.githubusercontent.com/TManzini/"
        "DebiasMulticlassWordEmbedding/master/Debiasing/data/vocab/"
    )
    WORD_SETS_FILES = [
        "gender_attributes_optm.json",
        "race_attributes_optm.json",
        "religion_attributes_optm.json",
    ]
    # fetch gender
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[0]) as file:
        gender = json.loads(file.read().decode())

        gender_definitional_sets = np.array(gender["definite_sets"])

        female_terms = gender_definitional_sets[:, 1].tolist()
        male_terms = gender_definitional_sets[:, 0].tolist()

        gender_analogy_templates = gender["analogy_templates"]["role"]
        male_roles = gender_analogy_templates["man"]
        female_roles = gender_analogy_templates["woman"]

        gender_eval_target = gender["eval_targets"]

    # fetch ethnicity
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[1]) as file:
        ethnicity = json.loads(file.read().decode())

        ethnicity_definitional_sets = np.array(ethnicity["definite_sets"])

        black = np.unique(ethnicity_definitional_sets[:, 0]).tolist()
        white = np.unique(ethnicity_definitional_sets[:, 1]).tolist()
        asian = np.unique(ethnicity_definitional_sets[:, 2]).tolist()

        ethnicity_analogy_templates = ethnicity["analogy_templates"]["role"]
        white_biased_words = ethnicity_analogy_templates["caucasian"]
        asian_biased_words = ethnicity_analogy_templates["asian"]
        black_biased_words = ethnicity_analogy_templates["black"]

        ethnicity_eval_target = ethnicity["eval_targets"]

    # fetch religion
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[2]) as file:
        religion = json.loads(file.read().decode())

        religion_definitional_sets = np.array(religion["definite_sets"])

        judaism = np.unique(religion_definitional_sets[:, 0]).tolist()
        christianity = np.unique(religion_definitional_sets[:, 1]).tolist()
        islam = np.unique(religion_definitional_sets[:, 2]).tolist()

        religion_analogy_templates = religion["analogy_templates"]["attribute"]
        greed = religion_analogy_templates["jew"]
        conservative = religion_analogy_templates["christian"]
        terrorism = religion_analogy_templates["muslim"]

        religion_eval_target = religion["eval_targets"]

    word_sets_dict = {
        # preprocessed word sets
        "male_terms": male_terms,
        "female_terms": female_terms,
        "male_roles": male_roles,
        "female_roles": female_roles,
        "black_terms": black,
        "white_terms": white,
        "asian_terms": asian,
        "black_biased_words": black_biased_words,
        "white_biased_words": white_biased_words,
        "asian_biased_words": asian_biased_words,
        "judaism_terms": judaism,
        "christianity_terms": christianity,
        "islam_terms": islam,
        "greed": greed,
        "conservative": conservative,
        "terrorism": terrorism,
        # original word sets
        "gender_definitional_sets": gender_definitional_sets.tolist(),
        "ethnicity_definitional_sets": ethnicity_definitional_sets.tolist(),
        "religion_definitional_sets": religion_definitional_sets.tolist(),
        "gender_analogy_templates": gender_analogy_templates,
        "ethnicity_analogy_templates": ethnicity_analogy_templates,
        "religion_analogy_templates": religion_analogy_templates,
        "gender_eval_target": gender_eval_target,
        "ethnicity_eval_target": ethnicity_eval_target,
        "religion_eval_target": religion_eval_target,
    }
    return word_sets_dict


def fetch_gn_glove() -> Dict[str, List[str]]:
    """Fetch the word sets used in the paper Learning Gender-Neutral Word Embeddings.

    This dataset contain two sets of 221 female and male related words.

    Reference
    ---------
    Zhao, J., Zhou, Y., Li, Z., Wang, W., & Chang, K. W. (2018, January). Learning
    Gender-Neutral Word Embeddings. In EMNLP.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary with male and female word sets.
    """

    BASE_URL = "https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/"
    FILES = [
        "female_word_file.txt",
        "male_word_file.txt",
    ]

    # fetch female words
    with urllib.request.urlopen(BASE_URL + FILES[0]) as file:
        female_terms = file.read().decode().split("\n")
        female_terms = list(filter(lambda x: x != "", female_terms))
    # fetch male words
    with urllib.request.urlopen(BASE_URL + FILES[1]) as file:
        male_terms = file.read().decode().split("\n")
        male_terms = list(filter(lambda x: x != "", male_terms))

    word_sets_dict = {
        "male_terms": male_terms,
        "female_terms": female_terms,
    }
    return word_sets_dict


def load_weat() -> Dict[str, List[str]]:
    """Load the word sets used in the paper *Semantics Derived Automatically*
    *From Language Corpora Contain Human-Like Biases*.
    It includes gender (male, female), ethnicity (black, white)
    and pleasant, unpleasant word sets, among others.

    References
    ----------

    Semantics derived automatically from language corpora contain human-like
    biases.
    Caliskan, A., Bryson, J. J., & Narayanan, A. (2017).
    Science, 356(6334), 183-186.

    Returns
    -------
    word_sets_dict : dict
        A dictionary with the word sets.


    """
    resource_package = __name__
    resource_path = "/".join(("data", "WEAT.json"))
    weat_data = pkg_resources.resource_string(resource_package, resource_path)

    data = json.loads(weat_data.decode())

    return data
