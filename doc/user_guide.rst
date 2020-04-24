.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

The following guide is designed to present the more general details about how 
to use the package. Below:

- First, we will present how to run a simple query using some embedding model. 
- Then how to run multiple queries on multiple embeddings.
- After that, how to compare the results of running multiple sets of queries 
on multiple embeddings using different metrics through ranking calculation.
- Finally, how to calculate the correlations between the rankings obtained.


Run a Query
===================================================================

The following code will show how to run a gender query using a word embedding
model as glove and the Word Embedding Association Test (WEAT) metric.

The common flow to perform a query in WEFE consist in three steps, 
which will be displayed next to the code:

>>> # Load the package
>>> from wefe.query import Query
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics.WEAT import WEAT
>>> from wefe.datasets.datasets import load_weat
>>> import gensim.downloader as api

1. Load the embedding models in a :code:`WordEmbeddingModel` object.

Load the Word Embedding pretrained model from gensim and then, create a 
:code:`WordEmbeddingModel` instance with it.
For this example, we will use a glove model of 25 dimensions trainer with a 
twitter dataset.

>>> twitter_25 = api.load('glove-twitter-25')
>>> model = WordEmbeddingModel(twitter_25, 'glove twitter dim=25')

2. Create the query usning a :code:`Query` object

Define the target and attribute sets from a  loaded, fetched or custom word 
sets and then, create a :code:`Query` that contains its. 

For this example, we will create a query with gender terms with respect to 
family and career.  The words we will use will be taken from the set of words
used in the WEAT paper (included in the package).

>>> # load the weat word sets
>>> word_sets = load_weat()
>>> 
>>> gender_query_1 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['career'], word_sets['family']],
>>>                        ['Male terms', 'Female terms'], ['Career', 'Family'])

3. Instance the Metric

Instance the metric that you will use and then, execute :code:`run_query` with the 
parameters created in the past steps. In this case we will use the 
:code:`WEAT` metric. 

>>> weat = WEAT()
>>> result = weat.run_query(query, model)
>>> print(result)
{'query_name': 'Male Terms and Female Terms wrt Arts and Science',
 'result': -0.010003209}

Run several Queries
===================

This package also implements a function that allows you to test several queries 
and word embedding models in one script.

The following code will show how to run various gender queries
over a different glove models trained using the twitter dataset. 
The queries will be executed using the WEAT variant, Effect size.

>>> from wefe.query import Query
>>> from wefe.datasets import load_weat
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics import WEAT, RNSB
>>> from wefe.utils import run_queries, plot_queries_results
>>> 
>>> import gensim.downloader as api

1. Load the models:

Load the glove twitter models. This models were trained using the same 
dataset, but varying only in the dimensions of the embeddings. 

>>> model_1 = WordEmbeddingModel(api.load('glove-twitter-25'),
>>>                              'glove twitter dim=25')
>>> model_2 = WordEmbeddingModel(api.load('glove-twitter-50'),
>>>                              'glove twitter dim=50')
>>> model_3 = WordEmbeddingModel(api.load('glove-twitter-100'),
>>>                              'glove twitter dim=100')
>>> 
>>> models = [model_1, model_2, model_3]

2. Load the word sets:

Now, we will load the WEAT word set. From this, we will create three 
queries that will intended to measure gender bias and two queries to measure 
ethnicity bias.

>>> # Load the WEAT word sets
>>> word_sets = load_weat()
>>> 
>>> # Create gender queries
>>> gender_query_1 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['career'], word_sets['family']],
>>>                        ['Male terms', 'Female terms'], ['Career', 'Family'])
>>> gender_query_2 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['science'], word_sets['arts']],
>>>                        ['Male terms', 'Female terms'], ['Science', 'Arts'])
>>> gender_query_3 = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>                        [word_sets['math'], word_sets['arts_2']],
>>>                        ['Male terms', 'Female terms'], ['Math', 'Arts'])
>>> 
>>> gender_queries = [gender_query_1, gender_query_2, gender_query_3]


3. Run the queries over all Word Embeddings using WEAT Effect Size. 

Now, to run our list of queries and models, we will use :code:`run_queries` function.
Its fundamental parameters are 3: it requires a metric, a list of queries 
and a list of embedding models. The name is optional.  

Note you can pass parameters to the metric using a dict in the 
:code:`metric_params` parameter. In this case, we specify that WEAT returns 
its Effect size variant as results.

>>> # Run the queries
>>> WEAT_gender_results = run_queries(WEAT,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   metric_params={'return_effect_size': True},
>>>                                   queries_set_name='Gender Queries')
>>> WEAT_gender_results


=====================  ===================================================  ==================================================  ===============================================
Model name               Male terms and Female terms wrt Career and Family    Male terms and Female terms wrt Science and Arts    Male terms and Female terms wrt Math and Arts
=====================  ===================================================  ==================================================  ===============================================
glove twitter dim=25                                              0.715369                                            0.766402                                         0.121468
glove twitter dim=50                                              0.799666                                           -0.660553                                        -0.589894
glove twitter dim=100                                             0.681933                                            0.641153                                        -0.399822
=====================  ===================================================  ==================================================  ===============================================

Important: In the event that a query loses more than 20% (by default) of words 
when converting one of its sets to embedding, the metric will return :code:`Nan`.
It behavior is also configurable by giving a float number to the parameter :code:`lost_vocabulary_threshold`. 

4. Plot the results in a barplot:

>>> # Plot the results
>>> plot_queries_results(WEAT_gender_results).show()


.. image:: images/WEAT_gender_results.png
  :alt: WEAT gender results


5. Aggregating Results:

When using run_queries, there is also the possibility of aggregate the 
results by embedding. To do this, you must first give the function the 
:code:`aggregate_results` parameter as :code:`True`. This default will activate
the option to aggregate the results by the average of their absolute values.

This aggregation function can be changed through the `aggregation_function`
parameter. Here you can specify a string that defines some of the aggregation 
types that are already implemented, as well as provide a function which 
operates on the dataframe of the results.

The default options available are:

- Average :code:`avg`
- Average of the absolute values :code:`abs_avg`
- Sum :code:`sum` 
- Sum of the absolute values, :code:`abs_sum`

For example, for the previous case, let's aggregate the results by the average of 
the absolute values obtained:

>>> WEAT_gender_results_agg = run_queries(WEAT,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   metric_params={'return_effect_size': True},
>>>                                   aggregate_results=True,
>>>                                   aggregation_function='abs_avg',
>>>                                   queries_set_name='Gender Queries')
>>> WEAT_gender_results_agg

=====================  ===================================================  ==================================================  ===============================================  ==================================================
model_name               Male terms and Female terms wrt Career and Family    Male terms and Female terms wrt Science and Arts    Male terms and Female terms wrt Math and Arts    WEAT: Gender Queries average of abs values score
=====================  ===================================================  ==================================================  ===============================================  ==================================================
glove twitter dim=25                                              0.715369                                            0.766402                                         0.121468                                            0.534413
glove twitter dim=50                                              0.799666                                           -0.660553                                        -0.589894                                            0.683371
glove twitter dim=100                                             0.681933                                            0.641153                                        -0.399822                                            0.574303
=====================  ===================================================  ==================================================  ===============================================  ==================================================

Finally, we can ask the function to return only the aggregated values 
(through :code:`return_only_aggregation` parameter) and then to plot them.

>>> WEAT_gender_results_agg = run_queries(WEAT,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   metric_params={'return_effect_size': True},
>>>                                   aggregate_results=True,
>>>                                   aggregation_function='abs_avg',
>>>                                   return_only_aggregation=True,
>>>                                   queries_set_name='Gender Queries')
>>> WEAT_gender_results_agg
>>> plot_queries_results(WEAT_gender_results_agg).show()


.. image:: images/WEAT_gender_results_agg.png
  :alt: WEAT gender results

Calculate Rankings
==================

When we want to measure various types of bias on different embedding models 
and different metrics, 2 big problems arise.

1. We do not want to lose or flatten the difference between the results of the 
various measured bias criteria. One type of bias can buffer or intensify another.

2. Metrics deliver their results on different scales, making them difficult 
to compare.

To show that, suppose we have two sets of queries: one that explores gender 
biases and one that explores ethnicity biases. Furthermore, we want to test 
these sets of queries on 3 glove models of 25, 50 and 100 dimensions trained 
using the same twitter corpus. In addition, we will use both WEAT and Relative 
Negative Sentiment Bias (RNSB) as metrics for the measurement.


1. Let's show the first problem: Lose or flatten the difference between the 
results of different bias criteria. 

We will execute the gender and ethnicity queries using WEAT and the 3 models
mentioned above. The results obtained are:

=====================  ==================================================  =====================================================
model_name               WEAT: Gender Queries average of abs values score    WEAT: Ethnicity Queries average of abs values score
=====================  ==================================================  =====================================================
glove twitter dim=25                                             0.210556                                                2.64632
glove twitter dim=50                                             0.292373                                                1.87431
glove twitter dim=100                                            0.225116                                                1.78469
=====================  ==================================================  =====================================================

As can be seen, the results of ethnicity bias are much greater than those
of gender.

2. For the second problem: Metrics deliver their results on different scales.

We will execute the gender queries using WEAT and RNSB metrics and the 3 models
mentioned above. The results obtained are:

=====================  ==================================================  ==================================================
model_name               WEAT: Gender Queries average of abs values score    RNSB: Gender Queries average of abs values score
=====================  ==================================================  ==================================================
glove twitter dim=25                                             0.210556                                           0.032673
glove twitter dim=50                                             0.292373                                           0.049429
glove twitter dim=100                                            0.225116                                           0.0312772
=====================  ==================================================  ==================================================

Now, we can see differences between the results of both metrics of an order 
of magnitude.

To solve both problems, we propose to create *rankings*. These allow us to 
compare more generally the scores of each embedding obtained by each of the 
tests without having to worry about the problems mentioned above.

Now, let's create rankings using the data used previously. The next code will 
load the models and create the queries: 

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
>>>                                   aggregate_results=True,
>>>                                   return_only_aggregation=True,
>>>                                   
>>>                                   queries_set_name='Gender Queries')
>>> 
>>> WEAT_ethnicity_results = run_queries(WEAT,
>>>                                      ethnicity_queries,
>>>                                      models,
>>>                                      aggregate_results=True,
>>>                                      return_only_aggregation=True,
>>>                                      queries_set_name='Ethnicity Queries')
>>>


>>> # Run the queries using RNSB
>>> RNSB_gender_results = run_queries(RNSB,
>>>                                   gender_queries,
>>>                                   models,
>>>                                   aggregate_results=True, 
>>>                                   return_only_aggregation=True,
>>>                                   queries_set_name='Gender Queries')
>>> 
>>> RNSB_ethnicity_results = run_queries(RNSB,
>>>                                      ethnicity_queries,
>>>                                      models,
>>>                                      aggregate_results=True,
>>>                                      return_only_aggregation=True,
>>>                                      queries_set_name='Ethnicity Queries')

   
To create the ranking, we will use :code:`create_ranking` function.
It takes all DataFrames with the calculated results and uses the 
last column (which assumes that it will find the scores already aggregated) to
create the rankings.


>>> ranking = create_ranking([
>>>     WEAT_gender_results, WEAT_ethnicity_results, RNSB_gender_results,
>>>     RNSB_ethnicity_results
>>> ])

=====================  ==================================================  =====================================================  ==================================================  =====================================================
model_name               WEAT: Gender Queries average of abs values score    WEAT: Ethnicity Queries average of abs values score    RNSB: Gender Queries average of abs values score    RNSB: Ethnicity Queries average of abs values score
=====================  ==================================================  =====================================================  ==================================================  =====================================================
glove twitter dim=25                                                    1                                                      3                                                   3                                                      3
glove twitter dim=50                                                    3                                                      2                                                   2                                                      1
glove twitter dim=100                                                   2                                                      1                                                   1                                                      2
=====================  ==================================================  =====================================================  ==================================================  =====================================================

Finally, we can plot those rankings using plot_ranking util. We have two options: 

1. With facet by Metric and Criteria:

This image shows the rankings separated by each bias criteria and metric (ie: by each column). 
Each bar represents the position of the embedding in the criteria-metric ranking.

>> plot_ranking(ranking, use_metric_as_facet=True)

.. image:: images/ranking_with_facet.png
  :alt: Ranking with facet

2. Without facet:

>> plot_ranking(ranking)

This image shows the accumulated rankings for each embeddings. 
Each bar represents the sum of the rankings obtained by each embedding. 
Each color inside a bar represent a different criteria-metric ranking.

.. image:: images/ranking_without_facet.png
  :alt: Ranking without facet


Ranking Correlations
====================

We can see how well the rankings obtained in the previous section relate using
a correlation matrix.
For this, we provide the function calculate_ranking_correlations. 
This takes as inputs the rankings and calculates the Spearman correlation
between them.

>>> from wefe.utils import calculate_ranking_correlations, plot_ranking_correlations
>>> correlations = calculate_ranking_correlations(ranking)
>>> correlations

===================================================  ==================================================  =====================================================  ==================================================  =====================================================
model                                                WEAT: Gender Queries average of abs values score    WEAT: Ethnicity Queries average of abs values score    RNSB: Gender Queries average of abs values score    RNSB: Ethnicity Queries average of abs values score
===================================================  ==================================================  =====================================================  ==================================================  =====================================================
WEAT: Gender Queries average of abs values score                                                    1                                                     -0.5                                                -0.5                                                   -1
WEAT: Ethnicity Queries average of abs values score                                                -0.5                                                    1                                                   1                                                      0.5
RNSB: Gender Queries average of abs values score                                                   -0.5                                                    1                                                   1                                                      0.5
RNSB: Ethnicity Queries average of abs values score                                                -1                                                      0.5                                                 0.5                                                    1
===================================================  ==================================================  =====================================================  ==================================================  =====================================================


Finally, we also provide a function to graph the correlations. 
This allows us to visually analyze in a very simple way how rankings relate to each other.


>>> correlation_fig = plot_ranking_correlations(correlations)
>>> correlation_fig.show()

.. image:: images/ranking_correlations.png
  :alt: Ranking without facet
