#############################################
Rank several embeddings on different criteria
#############################################

Load the queries
----------------

>>> from wefe.datasets import load_weat, fetch_eds, fetch_debias_multiclass, fetch_debiaswe, fetch_bingliu
>>> from wefe.query import Query
>>> 
>>> WEAT_wordsets = load_weat()
>>> RND_wordsets = fetch_eds()
>>> sentiments_wordsets = fetch_bingliu()
>>> debias_multiclass_wordsets = fetch_debias_multiclass()
>>> 
>>> # ----------------------------------------------------------------------------
>>> # Ethnicity Queries
>>> # ----------------------------------------------------------------------------
>>> 
>>> eth_1 = Query([RND_wordsets['names_white'], RND_wordsets['names_black']],
>>>               [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['White last names', 'Black last names'],
>>>               ['Pleasant', 'Unpleasant'])
>>> 
>>> eth_2 = Query([RND_wordsets['names_white'], RND_wordsets['names_asian']],
>>>               [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['White last names', 'Asian last names'],
>>>               ['Pleasant', 'Unpleasant'])
>>> 
>>> eth_3 = Query([RND_wordsets['names_white'], RND_wordsets['names_hispanic']],
>>>               [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['White last names', 'Hispanic last names'],
>>>               ['Pleasant', 'Unpleasant'])
>>> 
>>> eth_4 = Query(
>>>     [RND_wordsets['names_white'], RND_wordsets['names_black']],
>>>     [RND_wordsets['occupations_white'], RND_wordsets['occupations_black']],
>>>     ['White last names', 'Black last names'],
>>>     ['Occupations white', 'Occupations black'])
>>> 
>>> eth_5 = Query(
>>>     [RND_wordsets['names_white'], RND_wordsets['names_asian']],
>>>     [RND_wordsets['occupations_white'], RND_wordsets['occupations_asian']],
>>>     ['White last names', 'Asian last names'],
>>>     ['Occupations white', 'Occupations asian'])
>>> 
>>> eth_6 = Query(
>>>     [RND_wordsets['names_white'], RND_wordsets['names_hispanic']],
>>>     [RND_wordsets['occupations_white'], RND_wordsets['occupations_hispanic']],
>>>     ['White last names', 'Hispanic last names'],
>>>     ['Occupations white', 'Occupations hispanic'])
>>> 
>>> eth_sent_1 = Query([RND_wordsets['names_white'], RND_wordsets['names_black']],
>>>                    [
>>>                        sentiments_wordsets['positive_words'],
>>>                        sentiments_wordsets['negative_words']
>>>                    ], ['White last names', 'Black last names'],
>>>                    ['Positive words', 'Negative words'])
>>> 
>>> eth_sent_2 = Query([RND_wordsets['names_white'], RND_wordsets['names_asian']],
>>>                    [
>>>                        sentiments_wordsets['positive_words'],
>>>                        sentiments_wordsets['negative_words']
>>>                    ], ['White last names', 'Asian last names'],
>>>                    ['Positive words', 'Negative words'])
>>> 
>>> eth_sent_3 = Query(
>>>     [RND_wordsets['names_white'], RND_wordsets['names_hispanic']], [
>>>         sentiments_wordsets['positive_words'],
>>>         sentiments_wordsets['negative_words']
>>>     ], ['White last names', 'Hispanic last names'],
>>>     ['Positive words', 'Negative words'])
>>> 
>>> ethnicity_queries = [
>>>     eth_1, eth_2, eth_3, eth_4, eth_5, eth_6, eth_sent_1, eth_sent_2,
>>>     eth_sent_3
>>> ]
>>> 
>>> # ----------------------------------------------------------------------------
>>> # Gender Queries
>>> # ----------------------------------------------------------------------------
>>> 
>>> gender_1 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
>>>                  [WEAT_wordsets['career'], WEAT_wordsets['family']],
>>>                  ['Male terms', 'Female terms'], ['Career', 'Family'])
>>> 
>>> gender_2 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
>>>                  [WEAT_wordsets['math'], WEAT_wordsets['arts']],
>>>                  ['Male terms', 'Female terms'], ['Math', 'Arts'])
>>> 
>>> gender_3 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
>>>                  [WEAT_wordsets['science'], WEAT_wordsets['arts_2']],
>>>                  ['Male terms', 'Female terms'], ['Science', 'Arts'])
>>> 
>>> gender_4 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
>>>     RND_wordsets['adjectives_intelligence'],
>>>     RND_wordsets['adjectives_appearance']
>>> ], ['Male terms', 'Female terms'], ['Intelligence', 'Appearence'])
>>> 
>>> gender_5 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
>>>     RND_wordsets['adjectives_intelligence'],
>>>     RND_wordsets['adjectives_sensitive']
>>> ], ['Male terms', 'Female terms'], ['Intelligence', 'Sensitive'])
>>> 
>>> gender_6 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
>>>                  [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>                  ['Male terms', 'Female terms'], ['Pleasant', 'Unpleasant'])
>>> 
>>> gender_sent_1 = Query(
>>>     [RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
>>>         sentiments_wordsets['positive_words'],
>>>         sentiments_wordsets['negative_words']
>>>     ], ['Male terms', 'Female terms'], ['Positive words', 'Negative words'])
>>> 
>>> gender_role_1 = Query(
>>>     [RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
>>>         debias_multiclass_wordsets['male_roles'],
>>>         debias_multiclass_wordsets['female_roles']
>>>     ], ['Male terms', 'Female terms'], ['Man Roles', 'Woman Roles'])
>>> 
>>> gender_queries = [
>>>     gender_1, gender_2, gender_3, gender_4, gender_5, gender_sent_1,
>>>     gender_role_1
>>> ]
>>> 
>>> # ----------------------------------------------------------------------------
>>> # Religion Queries
>>> # ----------------------------------------------------------------------------
>>> 
>>> rel_1 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['islam_terms']
>>> ], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['Christianity terms', 'Islam terms'],
>>>               ['Pleasant', 'Unpleasant'])
>>> 
>>> rel_2 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['Christianity terms', 'Judaism terms'],
>>>               ['Pleasant', 'Unpleasant'])
>>> 
>>> rel_3 = Query([
>>>     debias_multiclass_wordsets['islam_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
>>>               ['Islam terms', 'Judaism terms'], ['Pleasant', 'Unpleasant'])
>>> 
>>> rel_4 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['islam_terms']
>>> ], [
>>>     debias_multiclass_wordsets['christian_related_words'],
>>>     debias_multiclass_wordsets['muslim_related_words']
>>> ], ['Christianity terms', 'Islam terms'],
>>>               ['Christian related words', 'Muslim related words'])
>>> 
>>> rel_5 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [
>>>     debias_multiclass_wordsets['christian_related_words'],
>>>     debias_multiclass_wordsets['jew_related_words']
>>> ], ['Christianity terms', 'Jew terms'],
>>>               ['Christian related words', 'Jew related words'])
>>> 
>>> rel_6 = Query([
>>>     debias_multiclass_wordsets['islam_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [
>>>     debias_multiclass_wordsets['muslim_related_words'],
>>>     debias_multiclass_wordsets['jew_related_words']
>>> ], ['Islam terms', 'Jew terms'], ['Musilm related words', 'Jew related words'])
>>> 
>>> rel_sent_1 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['islam_terms']
>>> ], [
>>>     sentiments_wordsets['positive_words'],
>>>     sentiments_wordsets['negative_words']
>>> ], ['Christianity terms', 'Islam terms'], ['Positive words', 'Negative words'])
>>> 
>>> rel_sent_2 = Query([
>>>     debias_multiclass_wordsets['christianity_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [
>>>     sentiments_wordsets['positive_words'],
>>>     sentiments_wordsets['negative_words']
>>> ], ['Christianity terms', 'Jew terms'], ['Positive words', 'Negative words'])
>>> 
>>> rel_sent_3 = Query([
>>>     debias_multiclass_wordsets['islam_terms'],
>>>     debias_multiclass_wordsets['judaism_terms']
>>> ], [
>>>     sentiments_wordsets['positive_words'],
>>>     sentiments_wordsets['negative_words']
>>> ], ['Islam terms', 'Jew terms'], ['Positive words', 'Negative words'])
>>> 
>>> religion_queries = [
>>>     rel_1, rel_2, rel_3, rel_4, rel_5, rel_6, rel_sent_1, rel_sent_2,
>>>     rel_sent_3
>>> ]

>>> queries_set_by_criteria = [[gender_queries, 'Gender'],
>>>                            [ethnicity_queries, 'Ethnicity'],
>>>                            [religion_queries, 'Religion']]