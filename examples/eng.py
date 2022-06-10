
import gensim.downloader as api
import numpy as np
from functools import reduce
import gensim.downloader as api
import os

from wefe.datasets import load_weat
from wefe.metrics import RNSB, WEAT, RND
from wefe.query import Query
from wefe.utils import run_queries
from wefe.word_embedding_model import WordEmbeddingModel
from flair.embeddings import WordEmbeddings
from wefe.utils import plot_queries_results
from wefe.utils import create_ranking
from wefe.utils import calculate_ranking_correlations, plot_ranking_correlations
from wefe.utils import plot_ranking

glove_embedding = WordEmbeddings('en')
glove_keyed_vectors = glove_embedding.precomputed_word_embeddings
model2 = WordEmbeddingModel(glove_keyed_vectors, 'en')

models = [model2]

# Load the WEAT word sets
word_sets = load_weat()

# Create gender queries
gender_query_1 = Query(
    [word_sets["male_terms"], word_sets["female_terms"]],
    [word_sets["career"], word_sets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
)

gender_query_2 = Query(
    [word_sets["male_terms"], word_sets["female_terms"]],
    [word_sets["science"], word_sets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
)

gender_query_3 = Query(
    [word_sets["male_terms"], word_sets["female_terms"]],
    [word_sets["math"], word_sets["arts_2"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts2"],
)


gender_queries = [gender_query_1, gender_query_2, gender_query_3]

weat = WEAT()

WEAT_gender_results = run_queries(
    WEAT,
    gender_queries,
    models,
    lost_vocabulary_threshold=0.3,
    metric_params={"preprocessors": [{"lowercase": True}]},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)

print(WEAT_gender_results)
# Plot the results
#plot_queries_results(WEAT_gender_results).show()
#WEAT_gender_results.to_csv('test.csv', mode='a', header=True, index=False)


# run the queries using WEAT effect size
WEAT_EZ_gender_results = run_queries(
    WEAT,
    gender_queries,
    models,
    lost_vocabulary_threshold=0.3,
    metric_params={"preprocessors": [{"lowercase": True}], "return_effect_size": True,},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)
print(WEAT_EZ_gender_results)
#plot_queries_results(WEAT_EZ_gender_results).show()
#WEAT_EZ_gender_results.to_csv('test.csv', mode='a', header=True, index=False)


RNSB_gender_results = run_queries(
    RNSB,
    gender_queries,
    models,
    lost_vocabulary_threshold=0.3,
    metric_params={"preprocessors": [{"lowercase": True}]},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)
print(RNSB_gender_results)
#plot_queries_results(RNSB_gender_results).show()
#RNSB_gender_results.to_csv('test.csv', mode='a', header=True, index=False)

RND_gender_results = run_queries(
    RND,
    gender_queries,
    models,
    metric_params={"preprocessors": [{}, {"lowercase": True, }], },
    queries_set_name="Gender Queries",
    aggregate_results=True,
    aggregation_function="abs_avg",
    generate_subqueries=True,
    warn_not_found_words=False,
)
print(RND_gender_results)
#plot_queries_results(RND_gender_results).show()
#RND_gender_results.to_csv('test.csv', mode='a', header=True, index=False)



