.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

The following guide is designed to present the more general details about how to use the package. Below:

- First, we will present how to run a simple query using some embedding model. 
- Then how to run multiple queries on multiple embeddings.
- After that, how to compare the results of running multiple sets of queries on multiple embeddings using different metrics through ranking calculation.
- Finally, how to calculate the correlations between the rankings obtained.


Run a Query
===================================================================

The following code will show how to run a query using one Word Embedding and a particular metric (WEAT).
The common flow to perform a query in WEFE consist in three steps, which will be displayed next to the code:

>>> from wefe.query import Query
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics.WEAT import WEAT
>>> from wefe.datasets.datasets import load_weat
>>> import gensim.downloader as api

1. Load the Word Embedding pretrained model from gensim and then, create a WordEmbeddingModel instance with it.
For this example, we will use a twitter_25 .

>>> twitter_25 = api.load('glove-twitter-25')
>>> model = WordEmbeddingModel(twitter_25, 'glove twitter dim=25')

2. Create the Query with a loaded, fetched or custom target and attribute word sets.
For this example, we will create a query with gender terms with respect to family and career. 
The words we will use will be taken from the set of words used in the WEAT paper (included in the package).

>>> # load the weat word sets
>>> word_sets = load_weat()
>>> 
>>> gender_query_1 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['career'], word_sets['family']],
>>>                        ['Male terms', 'Female terms'], ['Career', 'Family'])

3. Instance the metric that you will use and then, execute run_query with the parameters created in the past steps.
In this case we will use the WEAT metric. 

>>> weat = WEAT()
>>> result = weat.run_query(query, model)
>>> print(result)
{'query_name': 'Male Terms and Female Terms wrt Arts and Science',
 'result': -0.010003209}

Run several Queries
===================

This package also implements several scripts that allows you to run several queries on several word embedding models and 
The following code will show how to run various gender and ethnicity queries over a different glove models trained using the twitter dataset. 


>>> from wefe.query import Query
>>> from wefe.datasets import load_weat
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics import WEAT, RNSB
>>> from wefe.utils import run_queries, plot_queries_results
>>> 
>>> import gensim.downloader as api

1. Load the glove twitter models. This models were trained using the same dataset, but varying only in the dimensions of the embeddings. 

>>> model_1 = WordEmbeddingModel(api.load('glove-twitter-25'),
>>>                              'glove twitter dim=25')
>>> model_2 = WordEmbeddingModel(api.load('glove-twitter-50'),
>>>                              'glove twitter dim=50')
>>> model_3 = WordEmbeddingModel(api.load('glove-twitter-100'),
>>>                              'glove twitter dim=100')
>>> 
>>> models = [model_1, model_2, model_3]

2. Now, we will load the WEAT word set. From this, we will create three  queries that will intended to measure gender bias and two queries to measure ethnicity bias.

>>> # Load the WEAT word sets
>>> word_sets = load_weat()
>>> 
>>> # Create gender queries
>>> gender_query_1 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['career'], word_sets['family']],
>>>                        ['Male terms', 'Female terms'], ['Carrer', 'Family'])
>>> gender_query_2 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['science'], word_sets['arts']],
>>>                        ['Male terms', 'Female terms'], ['Science', 'Arts'])
>>> gender_query_3 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['math'], word_sets['arts_2']],
>>>                        ['Male terms', 'Female terms'], ['Math', 'Arts'])
>>> 
>>> gender_queries = [gender_query_1, gender_query_2, gender_query_3]


3. Run the queries over all Word Embeddings using WEAT Effect Size. 
To run a list of queries with a list of models, we will use run_queries function.
It requires a metric, a list of queries and a list of embedding models. The name is optional.  
Note you can pass parameters to the metric using metric_params parameter

>>> # Run the queries
>>> WEAT_gender_results = run_queries(WEAT,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   metric_params={'return_effect_size': True},
>>>                                   queries_set_name='Gender Queries')
>>> WEAT_gender_results


+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+
| Model                 |   Male terms and Female terms wrt Carrer and Family |   Male terms and Female terms wrt Science and Arts |   Male terms and Female terms wrt Math and Arts |   WEAT: Gender Queries average score |
+=======================+=====================================================+====================================================+=================================================+======================================+
| glove twitter dim=25  |                                            0.715369 |                                           0.766402 |                                        0.121468 |                             0.534413 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+
| glove twitter dim=50  |                                            0.799666 |                                          -0.660553 |                                       -0.589894 |                             0.683371 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+
| glove twitter dim=100 |                                            0.681933 |                                           0.641153 |                                       -0.399822 |                             0.574303 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+


Note that the last column brings the average of the scores obtained per row. 
These are calculated from the absolute values of the values shown. 
This behavior is configurable.

Important: In the event that a query loses more than 20% (by default) of words when converting one of its sets to embedding, the metric will return Nan.
It behavior is also configurable. 
In general, to calculate the averages, NaN are Ignored.

4. Plot the results in a barplot:

>>> # Plot the results
>>> WEAT_gender_ranking_fig = plot_queries_results(WEAT_gender_results)
>>> WEAT_gender_ranking_fig.show()


.. image:: images/WEAT_gender_results.png
  :alt: WEAT gender results




Calculate Rankings
==================

When we want to measure various types of bias on different embedding models and different metrics, 2 problems arise:

1. We don't want to lose the difference between the different bias criteria measured.
One type of bias may dampen or intensify another.

Results for Gender:

+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| Model                 |   Male terms and Female terms wrt Carrer and Family |   Male terms and Female terms wrt Science and Arts |   Male terms and Female terms wrt Math and Arts |   WEAT: Gender Queries average score | Model                 |
+=======================+=====================================================+====================================================+=================================================+======================================+=======================+
| glove twitter dim=25  |                                            0.715369 |                                           0.766402 |                                        0.121468 |                             0.534413 | glove twitter dim=25  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=50  |                                            0.799666 |                                          -0.660553 |                                       -0.589894 |                             0.683371 | glove twitter dim=50  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=100 |                                            0.681933 |                                           0.641153 |                                       -0.399822 |                             0.574303 | glove twitter dim=100 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+

Results for Ethnicity:

+-----------------------+----------------------------------------------------------------+--------------------------------------------------------------------+-----------------------------------------+-----------------------+
| Model                 |   European Names and African Names wrt Pleasant and Unpleasant |   European Names and African Names wrt Pleasant 2 and Unpleasant 2 |   WEAT: Ethnicity Queries average score | Model                 |
+=======================+================================================================+====================================================================+=========================================+=======================+
| glove twitter dim=25  |                                                        3.75292 |                                                            1.53973 |                                 2.64632 | glove twitter dim=25  |
+-----------------------+----------------------------------------------------------------+--------------------------------------------------------------------+-----------------------------------------+-----------------------+
| glove twitter dim=50  |                                                        2.56434 |                                                            1.18429 |                                 1.87431 | glove twitter dim=50  |
+-----------------------+----------------------------------------------------------------+--------------------------------------------------------------------+-----------------------------------------+-----------------------+
| glove twitter dim=100 |                                                        2.18871 |                                                            1.38067 |                                 1.78469 | glove twitter dim=100 |
+-----------------------+----------------------------------------------------------------+--------------------------------------------------------------------+-----------------------------------------+-----------------------+


2. Metrics deliver their results on different scales, making them difficult to compare.

Results for Gender on WEAT:

+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| Model                 |   Male terms and Female terms wrt Carrer and Family |   Male terms and Female terms wrt Science and Arts |   Male terms and Female terms wrt Math and Arts |   WEAT: Gender Queries average score | Model                 |
+=======================+=====================================================+====================================================+=================================================+======================================+=======================+
| glove twitter dim=25  |                                            0.715369 |                                           0.766402 |                                        0.121468 |                             0.534413 | glove twitter dim=25  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=50  |                                            0.799666 |                                          -0.660553 |                                       -0.589894 |                             0.683371 | glove twitter dim=50  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=100 |                                            0.681933 |                                           0.641153 |                                       -0.399822 |                             0.574303 | glove twitter dim=100 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+

Results for Gender on RNSB:

+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| Model                 |   Male terms and Female terms wrt Carrer and Family |   Male terms and Female terms wrt Science and Arts |   Male terms and Female terms wrt Math and Arts |   RNSB: Gender Queries average score | Model                 |
+=======================+=====================================================+====================================================+=================================================+======================================+=======================+
| glove twitter dim=25  |                                         0.000160544 |                                          0.192379  |                                     0.0145341   |                           0.0690244  | glove twitter dim=25  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=50  |                                         0.00730106  |                                          0.0175096 |                                     0.020789    |                           0.0151999  | glove twitter dim=50  |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+
| glove twitter dim=100 |                                         0.0134572   |                                          0.0035238 |                                     0.000843634 |                           0.00594154 | glove twitter dim=100 |
+-----------------------+-----------------------------------------------------+----------------------------------------------------+-------------------------------------------------+--------------------------------------+-----------------------+


To solve both problems, we propose to create rankings. 
For each evaluation we make (criteria, evaluation metrics) we create rankings of the performance of the embeddings.

The next code will load the models and create the queries: 

>>> from wefe.query import Query
>>> from wefe.datasets.datasets import load_weat
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics import WEAT, RNSB
>>> from wefe.utils import run_queries, create_ranking, plot_ranking, plot_ranking_correlations
>>> 
>>> import gensim.downloader as api
>>> 
>>> # Load the models
>>> model_1 = WordEmbeddingModel(api.load('glove-twitter-25'),
>>>                              'glove twitter dim=25')
>>> model_2 = WordEmbeddingModel(api.load('glove-twitter-50'),
>>>                              'glove twitter dim=50')
>>> model_3 = WordEmbeddingModel(api.load('glove-twitter-100'),
>>>                              'glove twitter dim=100')
>>> 
>>> models = [model_1, model_2, model_3]
>>> 
>>> 
>>> # Load the WEAT word sets
>>> word_sets = load_weat()
>>> 
>>> # Create gender queries
>>> gender_query_1 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['career'], word_sets['family']],
>>>                        ['Male terms', 'Female terms'], ['Carrer', 'Family'])
>>> gender_query_2 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['science'], word_sets['arts']],
>>>                        ['Male terms', 'Female terms'], ['Science', 'Arts'])
>>> gender_query_3 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['math'], word_sets['arts_2']],
>>>                        ['Male terms', 'Female terms'], ['Math', 'Arts'])
>>> 
>>> # Create ethnicity queries
>>> ethnicity_query_1 = Query([word_sets['european_american_names_5'],
>>>                            word_sets['african_american_names_5']],
>>>                           [word_sets['pleasant_5'], word_sets['unpleasant_5']],
>>>                           ['European Names', 'African Names'],
>>>                           ['Pleasant', 'Unpleasant'])
>>> 
>>> ethnicity_query_2 = Query([word_sets['european_american_names_7'],
>>>                            word_sets['african_american_names_7']], 
>>>                           [word_sets['pleasant_9'], word_sets['unpleasant_9']],
>>>                           ['European Names', 'African Names'],
>>>                           ['Pleasant 2', 'Unpleasant 2'])
>>> 
>>> gender_queries = [gender_query_1, gender_query_2, gender_query_3]
>>> ethnicity_queries = [ethnicity_query_1, ethnicity_query_2]


Now, we will run the queries with WEAT and RNSB:

>>> # Run the queries WEAT
>>> WEAT_gender_results = run_queries(WEAT,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   queries_set_name='Gender Queries')
>>> 
>>> WEAT_ethnicity_results = run_queries(WEAT,
>>>                                      ethnicity_queries,
>>>                                      models,
>>>                                      queries_set_name='Ethnicity Queries')


>>> # Run the queries using RNSB
>>> RNSB_gender_results = run_queries(RNSB,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   queries_set_name='Gender Queries')
>>> 
>>> RNSB_ethnicity_results = run_queries(RNSB,
>>>                                      ethnicity_queries,
>>>                                      models,
>>>                                      queries_set_name='Ethnicity Queries')

   
To create the ranking, we will use create_ranking util.
It takes all DataFrames with the previous calculated results and uses the average columns to create the rankings.
Note that all the results DataFrames must have the average columns. Otherwise, the function will raise a exception.

>>> ranking = create_ranking([
>>>     WEAT_gender_results, WEAT_ethnicity_results, RNSB_gender_results,
>>>     RNSB_ethnicity_results
>>> ])

+-----------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+-----------------------+
| Model                 |   WEAT: Gender Queries average score |   WEAT: Ethnicity Queries average score |   RNSB: Gender Queries average score |   RNSB: Ethnicity Queries average score | Model                 |
+=======================+======================================+=========================================+======================================+=========================================+=======================+
| glove twitter dim=25  |                                    1 |                                       3 |                                    3 |                                       3 | glove twitter dim=25  |
+-----------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+-----------------------+
| glove twitter dim=50  |                                    3 |                                       2 |                                    2 |                                       1 | glove twitter dim=50  |
+-----------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+-----------------------+
| glove twitter dim=100 |                                    2 |                                       1 |                                    1 |                                       2 | glove twitter dim=100 |
+-----------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+-----------------------+


Finally, we can plot those rankings using plot_ranking util. We have two options: 

1. With facet by Metric and Criteria:

This image shows the rankings separated by each bias criteria and metric (ie: by each column). 
Each bar represents the position of the embedding in the criteria-metric ranking.

.. image:: images/ranking_with_facet.png
  :alt: Ranking with facet

2. Without facet:

This image shows the accumulated rankings for each embeddings. 
Each bar represents the sum of the rankings obtained by each embedding. 
Each color inside a bar represent a different criteria-metric ranking.

.. image:: images/ranking_without_facet.png
  :alt: Ranking without facet


Ranking Correlations
====================

We can see how well the rankings obtained in the previous section relate using a correlation matrix.
For this, we provide the function calculate_ranking_correlations. 
This takes as inputs the rankings and calculates the Spearman correlation between them.

>>> from wefe.utils import calculate_ranking_correlations, plot_ranking_correlations
>>> correlations = calculate_ranking_correlations(ranking)
>>> correlations

+---------------------------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+---------------------------------------+
|                                       |   WEAT: Gender Queries average score |   WEAT: Ethnicity Queries average score |   RNSB: Gender Queries average score |   RNSB: Ethnicity Queries average score | Model                                 |
+=======================================+======================================+=========================================+======================================+=========================================+=======================================+
| WEAT: Gender Queries average score    |                                  1   |                                    -0.5 |                                 -1   |                                    -1   | WEAT: Gender Queries average score    |
+---------------------------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+---------------------------------------+
| WEAT: Ethnicity Queries average score |                                 -0.5 |                                     1   |                                  0.5 |                                     0.5 | WEAT: Ethnicity Queries average score |
+---------------------------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+---------------------------------------+
| RNSB: Gender Queries average score    |                                 -1   |                                     0.5 |                                  1   |                                     1   | RNSB: Gender Queries average score    |
+---------------------------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+---------------------------------------+
| RNSB: Ethnicity Queries average score |                                 -1   |                                     0.5 |                                  1   |                                     1   | RNSB: Ethnicity Queries average score |
+---------------------------------------+--------------------------------------+-----------------------------------------+--------------------------------------+-----------------------------------------+---------------------------------------+

Finally, we also provide a function to graph the correlations. 
This allows us to visually analyze in a very simple way how rankings relate to each other.


>>> correlation_fig = plot_ranking_correlations(correlations)
>>> correlation_fig.show()

.. image:: images/ranking_correlations.png
  :alt: Ranking without facet
