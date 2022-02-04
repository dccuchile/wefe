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

glove_embedding_fr = WordEmbeddings('fr')
glove_keyed_vectors_fr = glove_embedding_fr.precomputed_word_embeddings
modelfr = WordEmbeddingModel(glove_keyed_vectors_fr, 'fr')

models_fr = [modelfr]

# create the word sets
target_setsfr = [['Masculin', 'homme', 'garçon', 'frère', 'il', 'lui', 'le sien', 'fils'], ['femelle', 'femme', 'fille', 'sœur', 'elle', 'sa', 'la sienne', 'la fille']]
target_sets_namesfr = ['Male Terms', 'Female Terms']
attribute_setsfr = [['exécutif', 'la gestion', 'professionnel', 'société', 'un salaire', 'Bureau', 'Entreprise', 'carrière'], ['domicile', 'parents', 'enfants', 'famille', 'les cousins', 'mariage', 'mariage', 'es proches']]
attribute_sets_namesfr = ['career', 'family']
# create the query
gender_query_fr1 = Query(target_setsfr, attribute_setsfr, target_sets_namesfr, attribute_sets_namesfr)

# create the word sets
target_setsfr2 = [['Masculin', 'homme', 'garçon', 'frère', 'il', 'lui', 'le sien', 'fils'], ['femelle', 'femme', 'fille', 'sœur', 'elle', 'sa', 'la sienne', 'la fille']]
target_sets_namesfr2 = ['Male Terms', 'Female Terms']
attribute_setsfr2 = [['science', 'La technologie', 'la physique', 'chimie', 'Einstein', 'NASA', 'expérience', 'astronomie'], ['poésie', 'de l\'art', 'Danse', 'Littérature', 'roman', 'symphonie', 'drame', 'sculpture']]
attribute_sets_namesfr2 = ['Science', 'Arts']
# create the query
gender_query_fr2 = Query(target_setsfr2, attribute_setsfr2, target_sets_namesfr2, attribute_sets_namesfr2)

# create the word sets
target_setsfr3 = [['Masculin', 'homme', 'garçon', 'frère', 'il', 'lui', 'le sien', 'fils'], ['femelle', 'femme', 'fille', 'sœur', 'elle', 'sa', 'la sienne', 'la fille']]
target_sets_namesfr3 = ['Male Terms', 'Female Terms']
attribute_setsfr3 = [['math', 'algèbre', 'géométrie', 'calcul', 'équations', 'calcul', 'Nombres', 'une addition'], ['poésie', 'de l\'art', 'Shakespeare', 'Danse', 'Littérature', 'roman', 'symphonie', 'drame']]
attribute_sets_namesfr3 = ['Maths', 'Arts2']
# create the query
gender_query_fr3 = Query(target_setsfr3, attribute_setsfr3, target_sets_namesfr3, attribute_sets_namesfr3)

gender_queriesfr = [gender_query_fr1, gender_query_fr2, gender_query_fr3]


WEAT_gender_results_fr = run_queries(
    WEAT,
    gender_queriesfr,
    models_fr,
    lost_vocabulary_threshold=0.4,
    metric_params={"preprocessors": [{"lowercase": True}]},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)

print(WEAT_gender_results_fr)
# Plot the results
#plot_queries_results(WEAT_gender_results).show()
#WEAT_gender_results_fr.to_csv('test.csv', mode='a', header=True, index=False)


# run the queries using WEAT effect size
WEAT_EZ_gender_results_fr = run_queries(
    WEAT,
    gender_queriesfr,
    models_fr,
    lost_vocabulary_threshold=0.4,
    metric_params={"preprocessors": [{"lowercase": True}], "return_effect_size": True,},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)
print(WEAT_EZ_gender_results_fr)
#plot_queries_results(WEAT_EZ_gender_results).show()
#WEAT_EZ_gender_results_fr.to_csv('test.csv', mode='a', header=True, index=False)


RNSB_gender_results_fr = run_queries(
    RNSB,
    gender_queriesfr,
    models_fr,
    lost_vocabulary_threshold=0.4,
    metric_params={"preprocessors": [{"lowercase": True}]},
    aggregate_results=True,
    queries_set_name="Gender Queries",
)
print(RNSB_gender_results_fr)
#plot_queries_results(RNSB_gender_results).show()
#RNSB_gender_results_fr.to_csv('test.csv', mode='a', header=True, index=False)

RND_gender_results_fr = run_queries(
    RND,
    gender_queriesfr,
    models_fr,
    lost_vocabulary_threshold=0.4,
    metric_params={"preprocessors": [{}, {"lowercase": True, }], },
    queries_set_name="Gender Queries",
    aggregate_results=True,
    aggregation_function="abs_avg",
    generate_subqueries=True,
    warn_not_found_words=False,
)
print(RND_gender_results_fr)
#plot_queries_results(RND_gender_results_fr).show()
#RND_gender_results_fr.to_csv('test.csv', mode='a', header=True, index=False)

