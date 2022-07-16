import gensim.downloader as api
from flair.embeddings import WordEmbeddings

from wefe.datasets import load_weat
from wefe.metrics import RND, RNSB, WEAT
from wefe.query import Query
from wefe.utils import (calculate_ranking_correlations, create_ranking,
                        plot_queries_results, plot_ranking,
                        plot_ranking_correlations, run_queries)
from wefe.word_embedding_model import WordEmbeddingModel

glove_embedding = WordEmbeddings("nl")
glove_keyed_vectors = glove_embedding.precomputed_word_embeddings
model1 = WordEmbeddingModel(glove_keyed_vectors, "nl")

models = [model1]

# create the word sets
target_sets1 = [
    ["mannelijk", "Mens", "jongen", "broer", "hij", "hem", "zijn", "zoon"],
    ["vrouwelijk", "vrouw", "meisje", "zus", "ze", "haar", "haar", "dochter"],
]
target_sets_names1 = ["Male Terms", "Female Terms"]
attribute_sets1 = [
    [
        "leidinggevend",
        "beheer",
        "professioneel",
        "bedrijf",
        "salaris",
        "kantoor",
        "bedrijf",
        "carrière",
    ],
    [
        "huis",
        "ouders",
        "kinderen",
        "familie",
        "nichten en neven",
        "huwelijk",
        "bruiloft",
        "familieleden",
    ],
]
attribute_sets_names1 = ["career", "family"]
# create the query
gender_query_1 = Query(
    target_sets1, attribute_sets1, target_sets_names1, attribute_sets_names1
)

# create the word sets
target_sets2 = [
    ["mannelijk", "Mens", "jongen", "broer", "hij", "hem", "zijn", "zoon"],
    ["vrouwelijk", "vrouw", "meisje", "zus", "ze", "haar", "haar", "dochter"],
]
target_sets_names2 = ["Male Terms", "Female Terms"]
attribute_sets2 = [
    [
        "wetenschap",
        "technologie",
        "natuurkunde",
        "scheikunde",
        "Einstein",
        "NASA",
        "experiment",
        "astronomie",
    ],
    [
        "poëzie",
        "kunst",
        "dans",
        "literatuur",
        "roman",
        "symfonie",
        "drama",
        "beeldhouwwerk",
    ],
]
attribute_sets_names2 = ["Science", "Arts"]
# create the query
gender_query_2 = Query(
    target_sets2, attribute_sets2, target_sets_names2, attribute_sets_names2
)

# create the word sets
target_sets3 = [
    ["mannelijk", "Mens", "jongen", "broer", "hij", "hem", "zijn", "zoon"],
    ["vrouwelijk", "vrouw", "meisje", "zus", "ze", "haar", "haar", "dochter"],
]
target_sets_names3 = ["Male Terms", "Female Terms"]
attribute_sets3 = [
    [
        "wiskunde",
        "algebra",
        "geometrie",
        "rekening",
        "vergelijkingen",
        "berekening",
        "cijfers",
        "toevoeging",
    ],
    [
        "poëzie",
        "kunst",
        "Shakespeare",
        "dans",
        "literatuur",
        "roman",
        "symfonie",
        "drama",
    ],
]
attribute_sets_names3 = ["Maths", "Arts2"]
# create the query
gender_query_3 = Query(
    target_sets3, attribute_sets3, target_sets_names3, attribute_sets_names3
)

gender_queries = [gender_query_1, gender_query_2, gender_query_3]


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
# plot_queries_results(WEAT_gender_results).show()
# WEAT_gender_results.to_csv('test.csv', mode='a', header=True, index=False)


# run the queries using WEAT effect size
WEAT_EZ_gender_results = run_queries(
    WEAT,
    gender_queries,
    models,
    lost_vocabulary_threshold=0.3,
    metric_params={
        "preprocessors": [{"lowercase": True}],
        "return_effect_size": True,
    },
    aggregate_results=True,
    queries_set_name="Gender Queries",
)
print(WEAT_EZ_gender_results)
# plot_queries_results(WEAT_EZ_gender_results).show()
# WEAT_EZ_gender_results.to_csv('test.csv', mode='a', header=True, index=False)


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
# plot_queries_results(RNSB_gender_results).show()
# RNSB_gender_results.to_csv('test.csv', mode='a', header=True, index=False)

RND_gender_results = run_queries(
    RND,
    gender_queries,
    models,
    lost_vocabulary_threshold=0.3,
    metric_params={
        "preprocessors": [
            {},
            {
                "lowercase": True,
            },
        ],
    },
    queries_set_name="Gender Queries",
    aggregate_results=True,
    aggregation_function="abs_avg",
    generate_subqueries=True,
    warn_not_found_words=False,
)
print(RND_gender_results)
# plot_queries_results(RND_gender_results).show()
# RND_gender_results.to_csv('test.csv', mode='a', header=True, index=False)
