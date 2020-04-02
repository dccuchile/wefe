#!/usr/bin/env python
# coding: utf-8

# # WEFE Rankings Replication

# In[1]:

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# In[65]:

from wefe.datasets import load_weat, fetch_eds, fetch_debias_multiclass, fetch_debiaswe, fetch_bingliu
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics import WEAT, RNSB, RND
import gensim.downloader as api
from wefe.utils import run_queries, plot_queries_results, create_ranking, plot_ranking
from plotly.subplots import make_subplots
from IPython.core.display import display, HTML

# ## Load Queries

# In[4]:

WEAT_wordsets = load_weat()
RND_wordsets = fetch_eds()
sentiments_wordsets = fetch_bingliu()
debias_multiclass_wordsets = fetch_debias_multiclass()

# In[6]:

# ----------------------------------------------------------------------------
# Ethnicity Queries
# ----------------------------------------------------------------------------

eth_1 = Query([RND_wordsets['names_white'], RND_wordsets['names_black']],
              [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['White last names', 'Black last names'],
              ['Pleasant', 'Unpleasant'])

eth_2 = Query([RND_wordsets['names_white'], RND_wordsets['names_asian']],
              [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['White last names', 'Asian last names'],
              ['Pleasant', 'Unpleasant'])

eth_3 = Query([RND_wordsets['names_white'], RND_wordsets['names_hispanic']],
              [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['White last names', 'Hispanic last names'],
              ['Pleasant', 'Unpleasant'])

eth_4 = Query(
    [RND_wordsets['names_white'], RND_wordsets['names_black']],
    [RND_wordsets['occupations_white'], RND_wordsets['occupations_black']],
    ['White last names', 'Black last names'],
    ['Occupations white', 'Occupations black'])

eth_5 = Query(
    [RND_wordsets['names_white'], RND_wordsets['names_asian']],
    [RND_wordsets['occupations_white'], RND_wordsets['occupations_asian']],
    ['White last names', 'Asian last names'],
    ['Occupations white', 'Occupations asian'])

eth_6 = Query(
    [RND_wordsets['names_white'], RND_wordsets['names_hispanic']],
    [RND_wordsets['occupations_white'], RND_wordsets['occupations_hispanic']],
    ['White last names', 'Hispanic last names'],
    ['Occupations white', 'Occupations hispanic'])

eth_sent_1 = Query([RND_wordsets['names_white'], RND_wordsets['names_black']],
                   [
                       sentiments_wordsets['positive_words'],
                       sentiments_wordsets['negative_words']
                   ], ['White last names', 'Black last names'],
                   ['Positive words', 'Negative words'])

eth_sent_2 = Query([RND_wordsets['names_white'], RND_wordsets['names_asian']],
                   [
                       sentiments_wordsets['positive_words'],
                       sentiments_wordsets['negative_words']
                   ], ['White last names', 'Asian last names'],
                   ['Positive words', 'Negative words'])

eth_sent_3 = Query(
    [RND_wordsets['names_white'], RND_wordsets['names_hispanic']], [
        sentiments_wordsets['positive_words'],
        sentiments_wordsets['negative_words']
    ], ['White last names', 'Hispanic last names'],
    ['Positive words', 'Negative words'])

ethnicity_queries = [
    eth_1, eth_2, eth_3, eth_4, eth_5, eth_6, eth_sent_1, eth_sent_2,
    eth_sent_3
]

# In[7]:

# ----------------------------------------------------------------------------
# Gender Queries
# ----------------------------------------------------------------------------

gender_1 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
                 [WEAT_wordsets['career'], WEAT_wordsets['family']],
                 ['Male terms', 'Female terms'], ['Career', 'Family'])

gender_2 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
                 [WEAT_wordsets['math'], WEAT_wordsets['arts']],
                 ['Male terms', 'Female terms'], ['Math', 'Arts'])

gender_3 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
                 [WEAT_wordsets['science'], WEAT_wordsets['arts_2']],
                 ['Male terms', 'Female terms'], ['Science', 'Arts'])

gender_4 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
    RND_wordsets['adjectives_intelligence'],
    RND_wordsets['adjectives_appearance']
], ['Male terms', 'Female terms'], ['Intelligence', 'Appearence'])

gender_5 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
    RND_wordsets['adjectives_intelligence'],
    RND_wordsets['adjectives_sensitive']
], ['Male terms', 'Female terms'], ['Intelligence', 'Sensitive'])

gender_6 = Query([RND_wordsets['male_terms'], RND_wordsets['female_terms']],
                 [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
                 ['Male terms', 'Female terms'], ['Pleasant', 'Unpleasant'])

gender_sent_1 = Query(
    [RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
        sentiments_wordsets['positive_words'],
        sentiments_wordsets['negative_words']
    ], ['Male terms', 'Female terms'], ['Positive words', 'Negative words'])

gender_role_1 = Query(
    [RND_wordsets['male_terms'], RND_wordsets['female_terms']], [
        debias_multiclass_wordsets['male_roles'],
        debias_multiclass_wordsets['female_roles']
    ], ['Male terms', 'Female terms'], ['Man Roles', 'Woman Roles'])

gender_queries = [
    gender_1, gender_2, gender_3, gender_4, gender_5, gender_sent_1,
    gender_role_1
]

# In[8]:

# ----------------------------------------------------------------------------
# Religion Queries
# ----------------------------------------------------------------------------

rel_1 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['islam_terms']
], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['Christianity terms', 'Islam terms'],
              ['Pleasant', 'Unpleasant'])

rel_2 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['Christianity terms', 'Judaism terms'],
              ['Pleasant', 'Unpleasant'])

rel_3 = Query([
    debias_multiclass_wordsets['islam_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [WEAT_wordsets['pleasant_5'], WEAT_wordsets['unpleasant_5']],
              ['Islam terms', 'Judaism terms'], ['Pleasant', 'Unpleasant'])

rel_4 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['islam_terms']
], [
    debias_multiclass_wordsets['christian_related_words'],
    debias_multiclass_wordsets['muslim_related_words']
], ['Christianity terms', 'Islam terms'],
              ['Christian related words', 'Muslim related words'])

rel_5 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [
    debias_multiclass_wordsets['christian_related_words'],
    debias_multiclass_wordsets['jew_related_words']
], ['Christianity terms', 'Jew terms'],
              ['Christian related words', 'Jew related words'])

rel_6 = Query([
    debias_multiclass_wordsets['islam_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [
    debias_multiclass_wordsets['muslim_related_words'],
    debias_multiclass_wordsets['jew_related_words']
], ['Islam terms', 'Jew terms'], ['Musilm related words', 'Jew related words'])

rel_sent_1 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['islam_terms']
], [
    sentiments_wordsets['positive_words'],
    sentiments_wordsets['negative_words']
], ['Christianity terms', 'Islam terms'], ['Positive words', 'Negative words'])

rel_sent_2 = Query([
    debias_multiclass_wordsets['christianity_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [
    sentiments_wordsets['positive_words'],
    sentiments_wordsets['negative_words']
], ['Christianity terms', 'Jew terms'], ['Positive words', 'Negative words'])

rel_sent_3 = Query([
    debias_multiclass_wordsets['islam_terms'],
    debias_multiclass_wordsets['judaism_terms']
], [
    sentiments_wordsets['positive_words'],
    sentiments_wordsets['negative_words']
], ['Islam terms', 'Jew terms'], ['Positive words', 'Negative words'])

religion_queries = [
    rel_1, rel_2, rel_3, rel_4, rel_5, rel_6, rel_sent_1, rel_sent_2,
    rel_sent_3
]

# In[9]:
queries_sets = [[gender_queries, 'Gender'], [ethnicity_queries, 'Ethnicity'],
                [religion_queries, 'Religion']]

# ## Load models

# In[10]:

glove_twitter_25 = WordEmbeddingModel(api.load("glove-twitter-25"),
                                      "glove-twitter-25")
glove_twitter_50 = WordEmbeddingModel(api.load("glove-twitter-50"),
                                      "glove-twitter-50")
glove_twitter_100 = WordEmbeddingModel(api.load("glove-twitter-100"),
                                       "glove-twitter-100")
glove_twitter_200 = WordEmbeddingModel(api.load("glove-twitter-200"),
                                       "glove-twitter-200")

# In[11]:

models = [
    glove_twitter_25, glove_twitter_50, glove_twitter_100, glove_twitter_200
]

# ## Runners by metric

# In[83]:


def evaluate_WEAT(queries_set, models_arr):
    return run_queries(WEAT, queries_set[0], models_arr,
                       queries_set_name=queries_set[1],
                       include_average_by_embedding='include')


def evaluate_WEAT_effect_size(queries_set, models_arr):
    return run_queries(WEAT, queries_set[0], models_arr,
                       queries_set_name=queries_set[1],
                       metric_params={'return_effect_size': True},
                       include_average_by_embedding='include')


def evaluate_RND(queries_set, models_arr):
    subqueries = []
    for query in queries_set[0]:
        subqueries += query.get_subqueries((2, 1))

    return run_queries(RND, subqueries, models_arr,
                       queries_set_name=queries_set[1],
                       include_average_by_embedding='include')


RNSB_NUM_ITERS = 30


def evaluate_RNSB(queries_set, models_arr):

    RNSB_scores_iter = []
    ommited = 0

    # run several times the metric to calculate the avg scores.
    # it avoids outliers.
    for i in range(RNSB_NUM_ITERS):
        try:
            RNSB_scores_iter.append(
                run_queries(RNSB, queries_set[0], models_arr,
                            queries_set_name=queries_set[1],
                            include_average_by_embedding='include'))
        except Exception as e:
            ommited += 1
    if ommited != 0:
        print('\tIterations ommited: {}'.format(ommited))
    RNSB_scores = reduce(
        (lambda x, y: x + y), RNSB_scores_iter) / RNSB_NUM_ITERS

    return RNSB_scores


runners = [[evaluate_WEAT, 'WEAT'], [evaluate_WEAT_effect_size, 'WEAT_EZ'],
           [evaluate_RND, 'RND'], [evaluate_RNSB, 'RNSB']]

# In[87]:


def run_all(metric_runner, queries_sets, models):

    results = []
    for queries_set in queries_sets:
        results.append(metric_runner[0](queries_set, models))

    results_plot = make_subplots(rows=3, cols=1, vertical_spacing=0.2,
                                 subplot_titles=[q[1] for q in queries_sets])

    for row, queries_results in enumerate(results):
        fig_i = plot_queries_results(queries_results, )
        for trace in fig_i['data']:
            results_plot.add_trace(trace, row=row + 1, col=1)

    results_plot.update_layout(
        showlegend=True,
        title_text="{} results for each query set".format(metric_runner[1]),
        height=1200)
    results_plot.show()

    ranking = create_ranking(results)

    display(HTML(ranking.to_html()))
    ranking_plot = plot_ranking(ranking, use_metric_as_facet=False)

    return results, ranking, results_plot, ranking_plot


# In[ ]:

all_stuff = []
for runner in runners:
    all_stuff.append(run_all(runner, queries_sets, models))

# In[78]:

import pandas as pd

ranking_general = pd.DataFrame()

for r in all_stuff:
    ranking_general = pd.concat([ranking_general, r[1]], axis=1)
ranking_general

# In[80]:

ranking_general_plot = plot_ranking(ranking_general)

# In[ ]:

# In[81]:

import os

if not os.path.exists('./results'):
    os.mkdir('./results')

for r, metric in zip(all_stuff, runners):
    for idx, criteria in enumerate(queries_sets):
        r[0][idx].to_csv('./results/results_{}_{}.csv'.format(
            metric[1], criteria[1]))
    r[1].to_csv('./results/rankings_{}.csv'.format(metric[1]))
    r[2].write_image('./results/results_{}.svg'.format(metric[1]))
    r[3].write_image('./results/rankings_{}.svg'.format(metric[1]))

ranking_general_plot.write_image('./results/ranking_general_plot.svg')
