import gensim.downloader as api

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

glove_embedding = WordEmbeddings('es')
glove_keyed_vectors = glove_embedding.precomputed_word_embeddings
model1 = WordEmbeddingModel(glove_keyed_vectors, 'es')

models = [model1]

# create the word sets
target_sets1 = [['masculino', 'hombre', 'chico', 'hermano', 'él', 'él', 'su', 'hijo'], ['mujer', 'mujer', 'muchacha', 'hermana', 'ella', 'ella', 'suyo', 'hija']]
target_sets_names1 = ['Male Terms', 'Female Terms']
attribute_sets1 = [['ejecutivo', 'administración', 'profesional', 'corporación', 'salario', 'oficina', 'negocio', 'carrera profesional'], ['hogar', 'padres', 'niños', 'familia', 'primos', 'matrimonio', 'boda', 'parientes']]
attribute_sets_names1 = ['career', 'family']
# create the query
gender_query_1 = Query(target_sets1, attribute_sets1, target_sets_names1, attribute_sets_names1)

# create the word sets
target_sets2 = [['masculino', 'hombre', 'chico', 'hermano', 'él', 'él', 'su', 'hijo'], ['mujer', 'mujer', 'muchacha', 'hermana', 'ella', 'ella', 'suyo', 'hija']]
target_sets_names2 = ['Male Terms', 'Female Terms']
attribute_sets2 = [['Ciencias', 'tecnología', 'física', 'química', 'Einstein', 'NASA', 'experimentar', 'astronomía'], ['poesía',  'Arte', 'danza', 'literatura', 'novela', 'sinfonía', 'drama','escultura']]
attribute_sets_names2 = ['Science', 'Arts']
# create the query
gender_query_2 = Query(target_sets2, attribute_sets2, target_sets_names2, attribute_sets_names2)

# create the word sets
target_sets3 = [['masculino', 'hombre', 'chico', 'hermano', 'él', 'él', 'su', 'hijo'], ['mujer', 'mujer', 'muchacha', 'hermana', 'ella', 'ella', 'suyo', 'hija']]
target_sets_names3 = ['Male Terms', 'Female Terms']
attribute_sets3 = [['Matemáticas', 'álgebra', 'geometría', 'cálculo', 'ecuaciones', 'cálculo', 'números', 'adición'], ['poesía', 'Arte', 'Shakespeare', 'danza', 'literatura', 'novela', 'sinfonía', 'drama']]
attribute_sets_names3 = ['Maths', 'Arts2']
# create the query
gender_query_3 = Query(target_sets3, attribute_sets3, target_sets_names3, attribute_sets_names3)

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
    lost_vocabulary_threshold=0.3,
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
