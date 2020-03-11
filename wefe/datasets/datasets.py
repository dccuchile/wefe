import pandas as pd
import urllib.request
import patoolib
import shutil
import os
import json
import numpy as np


def fetch_eds(occupations_year: int = 2015):
    """Fetch the word sets used in the experiments of Word embeddings quantify 100 years of gender and ethnic stereotypes paper. 
    
    Parameters
    ----------
    occupations_year : int, optional
        The year of the census for the occupations file. 
        Available years: {'1850', '1860', '1870', '1880', '1900', '1910', '1920', '1930', '1940', '1950', '1960', 
        '1970', '1980', '1990', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', 
        '2010', '2011', '2012', '2013', '2014', '2015'}, by default 2015
    
    Returns
    -------
    word_sets_dict : dict
        A dictionary with the word sets.

    References
    ----------
    Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. 
    Proceedings of the National Academy of Sciences, 115(16), E3635-E3644.
    """

    EDS_BASE_URL = 'https://raw.githubusercontent.com/nikhgarg/EmbeddingDynamicStereotypes/master/data/'
    EDS_WORD_SETS_NAMES = [
        'adjectives_appearance.txt',
        'adjectives_intelligencegeneral.txt',
        'adjectives_otherization.txt',
        'adjectives_sensitive.txt',
        'female_pairs.txt',
        'male_pairs.txt',
        'names_asian.txt',
        'names_black.txt',
        'names_chinese.txt',
        'names_hispanic.txt',
        'names_russian.txt',
        'names_white.txt',
        'words_christianity.txt',
        'words_islam.txt',
        'words_terrorism.txt',
    ]

    # ---- Word sets ----

    # read the word sets from the source.
    word_sets = []
    for EDS_words_set_name in EDS_WORD_SETS_NAMES:
        name = EDS_words_set_name.replace('.txt', '').replace('_', ' ').capitalize()
        word_sets.append(pd.read_csv(EDS_BASE_URL + EDS_words_set_name, names=[name]))

    word_sets_dict = pd.concat(word_sets, sort=False, axis=1).to_dict(orient='list')

    # turn the dataframe into a python dict without nan.
    for dataset_name in word_sets_dict:
        word_sets_dict[dataset_name] = list(filter(lambda x: not pd.isnull(x), word_sets_dict[dataset_name]))

    # ---- Occupations by Gender ----

    # fetch occupations by gender
    gender_occupations = pd.read_csv(EDS_BASE_URL + 'occupation_percentages_gender_occ1950.csv')
    # filter by year
    gender_occupations_year_filtered = gender_occupations[gender_occupations['Census year'] == occupations_year]

    # get male occupations
    male_occupations = gender_occupations_year_filtered[
        gender_occupations_year_filtered['Male'] >= gender_occupations_year_filtered['Female']]
    male_occupations = male_occupations['Occupation'].values.tolist()

    # get female occupations
    female_occupations = gender_occupations_year_filtered[
        gender_occupations_year_filtered['Male'] < gender_occupations_year_filtered['Female']]
    female_occupations = female_occupations['Occupation'].values.tolist()

    word_sets_dict['Male Occupations'] = male_occupations
    word_sets_dict['Female Occupations'] = female_occupations

    word_sets_dict['Male terms'] = word_sets_dict.pop('Male pairs')
    word_sets_dict['Female terms'] = word_sets_dict.pop('Female pairs')
    word_sets_dict['Adjectives intelligence'] = word_sets_dict.pop('Adjectives intelligencegeneral')

    return word_sets_dict


def fetch_debiaswe():
    """[summary]
    
    Returns
    -------
    [type]
        [description]
    """

    DEBIAS_WE_BASE_URL = 'https://raw.githubusercontent.com/tolga-b/debiaswe/master/data/'
    DEBIAS_WE_WORD_SETS = ['definitional_pairs.json', 'equalize_pairs.json', 'professions.json']

    male_female_words = pd.read_json(DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[0])
    male_words = male_female_words[[0]].values.flatten().tolist()
    female_words = male_female_words[[1]].values.flatten().tolist()

    pairs = pd.read_json(DEBIAS_WE_BASE_URL + DEBIAS_WE_WORD_SETS[1])
    male_related_words = pairs[0].str.lower().values.flatten().tolist()
    female_related_words = pairs[1].str.lower().values.flatten().tolist()

    word_sets_dict = {
        'Male terms': male_words,
        'Female terms': female_words,
        'Male related words': male_related_words,
        'Female related words': female_related_words
    }

    return word_sets_dict


def fetch_bingliu():
    """Fetch the bing-liu sentiment lexicon form the source. 
    
    Returns
    -------
    dict
        A dictionary with the positive and negative words.

    References
    -------
    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
    Proceedings of the ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
    Washington, USA, 
    """

    # download the file
    if not os.path.exists('./lexicon.rar'):
        print('Fetching file...')
        url = 'http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar'
        local_fname = './lexicon.rar'
        urllib.request.urlretrieve(url, local_fname)
    else:
        print('Lexicon file already downloaded')

    # extract the file
    patoolib.extract_archive("./lexicon.rar", outdir="./")

    # load and clean the word sets
    negative_word_set_path = './opinion-lexicon-English/negative-words.txt'
    positive_word_set_path = './opinion-lexicon-English/positive-words.txt'

    negative = pd.read_csv(negative_word_set_path, sep='\n', header=None, names=['word'], encoding='latin-1')
    negative_cleaned = negative.loc[30:,].values.flatten().tolist()
    positive = pd.read_csv(positive_word_set_path, sep='\n', header=None, names=['word'], encoding='latin-1')
    positive_cleaned = positive.loc[29:,].values.flatten().tolist()

    bingliu_lexicon = {'Positive words': positive_cleaned, 'Negative words': negative_cleaned}

    # cleanup
    try:
        shutil.rmtree('./opinion-lexicon-English')
        os.remove("./lexicon.rar")
    except Exception as error:
        print('Unable to perform the cleanup of lexicon files: {}'.format(error))

    return bingliu_lexicon


def fetch_debias_multiclass():
    """[summary]
    
    Returns
    -------
    [type]
        [description]

    References
    ------
    Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings
    Thomas Manzini, Yao Chong Lim, Yulia Tsvetkov, Alan W Black
    """

    BASE_URL = 'https://raw.githubusercontent.com/TManzini/DebiasMulticlassWordEmbedding/master/Debiasing/data/vocab/'
    WORD_SETS_FILES = ['gender_attributes_optm.json', 'race_attributes_optm.json', 'religion_attributes_optm.json']
    # fetch gender
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[0]) as file:
        gender = json.loads(file.read().decode())

        gender_terms = np.array(gender['definite_sets'])
        female_terms = gender_terms[:, 1].tolist()
        male_terms = gender_terms[:, 0].tolist()

        gender_related_words = gender['analogy_templates']['role']
        male_related_words = gender_related_words['man']
        female_related_words = gender_related_words['woman']
    # fetch race
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[1]) as file:
        race = json.loads(file.read().decode())

        race_terms = np.array(race['definite_sets'])
        black = np.unique(race_terms[:, 0]).tolist()
        white = np.unique(race_terms[:, 1]).tolist()
        asian = np.unique(race_terms[:, 2]).tolist()

        race_related_words = race['analogy_templates']['role']
        white_related_words = race_related_words['caucasian']
        asian_related_words = race_related_words['asian']
        black_related_words = race_related_words['black']
    # fetch religion
    with urllib.request.urlopen(BASE_URL + WORD_SETS_FILES[2]) as file:
        religion = json.loads(file.read().decode())

        religion_terms = np.array(religion['definite_sets'])
        judaism = np.unique(religion_terms[:, 0]).tolist()
        christianity = np.unique(religion_terms[:, 1]).tolist()
        islam = np.unique(religion_terms[:, 2]).tolist()

        religion_related_words = religion['analogy_templates']['attribute']
        judaism_related_words = religion_related_words['jew']
        christianity_related_words = religion_related_words['christian']
        islam_related_words = religion_related_words['muslim']

    word_sets_dict = {
        'Male terms': male_terms,
        'Female terms': female_terms,
        'Male related words': male_related_words,
        'Female related words': female_related_words,
        'Black terms': black,
        'White terms': white,
        'Asian terms': asian,
        'Black related words': black_related_words,
        'White related words': white_related_words,
        'Asian related words': asian_related_words,
        'Judaism terms': judaism,
        'Christianity terms': christianity,
        'Islam terms': islam,
        'Jew related words': judaism_related_words,
        'Christian related words': christianity_related_words,
        'Muslim related words': islam_related_words,
    }
    return word_sets_dict


def load_weat():
    """Loads the datasets used in the Semantics derived automatically from language corpora contain human-like biases paper tests. 

    Returns
    -------
    word_sets_dict : dict
        A dictionary with the word sets.

    References
    ----------
    Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186.

    """
    with open('./wefe/datasets/WEAT.json') as WEAT_json:
        word_sets_dict_uncleaned = json.load(WEAT_json)
        word_sets_dict = {}
        for key in word_sets_dict_uncleaned:
            new_key = key.replace('_', ' ').capitalize()
            word_sets_dict[new_key] = word_sets_dict_uncleaned[key]

        WEAT_json.close()

    return word_sets_dict
