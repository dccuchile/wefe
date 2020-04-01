.. title:: User guide : contents

.. _user_guide:

==================================================
User guide
==================================================


Run a Query
===================================================================

The following code will show how to run aquery using one Word Embedding and a particular metric (WEAT).
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
>>> gender_query_1 = Query([word_sets['Male terms'], word_sets['Female terms']],
>>>                        [word_sets['Career'], word_sets['Family']],
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
>>> from wefe.datasets.datasets import load_weat
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics.WEAT import WEAT
>>> from wefe.utils import run_queries
>>> import gensim.downloader as api

1. Load the glove twitter models. This models were trained using the same dataset, but varying only in the dimensions of the embeddings. 

>>> model_1 = WordEmbeddingModel(api.load('glove-twitter-25'), 'glove twitter dim=25')
>>> model_2 = WordEmbeddingModel(api.load('glove-twitter-50'), 'glove twitter dim=50')
>>> model_3 = WordEmbeddingModel(api.load('glove-twitter-100'), 'glove twitter dim=100')
>>> models = [model_1, model_2, model_3]

2. Now, we will load the WEAT word set. From this, we will create three  queries that will intended to measure gender bias and two queries to measure ethnicity bias.

>>> # load the WEAT word set (included into the package)
>>> word_sets = load_weat()
>>> 
>>> # create the gender queries.
>>> gender_query_1 = Query([word_sets['Male terms'], word_sets['Female terms']],
>>>                        [word_sets['Career'], word_sets['Family']],
>>>                        ['Male terms', 'Female terms'], ['Carrer', 'Family'])
>>> gender_query_2 = Query([word_sets['Male terms'], word_sets['Female terms']],
>>>                        [word_sets['Science'], word_sets['Arts']],
>>>                        ['Male terms', 'Female terms'], ['Science', 'Arts'])
>>> gender_query_3 = Query([word_sets['Male terms'], word_sets['Female terms']],
>>>                        [word_sets['Math'], word_sets['Arts 2']],
>>>                        ['Male terms', 'Female terms'], ['Math', 'Arts'])
>>>
>>> # create the ethnicity queries.
>>> ethnicity_query_1 = Query([word_sets['European american names 5'],
>>>                            word_sets['African american names 5']],
>>>                           [word_sets['Pleasant 5'], word_sets['Unpleasant 5']],
>>>                           ['European Names', 'African Names'],
>>>                           ['Pleasant', 'Unpleasant'])
>>> 
>>> ethnicity_query_2 = Query([word_sets['European american names 7'],
>>>                            word_sets['African american names 7']], 
>>>                           [word_sets['Pleasant 9'], word_sets['Unpleasant 9']],
>>>                           ['European Names', 'African Names'],
>>>                           ['Pleasant 2', 'Unpleasant 2'])
>>>
>>> queries = [gender_query_1, gender_query_2, gender_query_3, ethnicity_query_1, ethnicity_query_2]

3. Run the queries over all Word Embeddings using WEAT:

>>> WEAT_queries_results = run_queries(WEAT,
>>>                       queries,
>>>                       models,
>>>                       queries_set_name='Gender and ethnicity Queries')

+------------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| Model name - Queries   | Male terms and Female terms wrt Career and Family  | Male terms and Female terms wrt Science and Arts  | Male terms and Female terms wrt Math and Arts  | European Names and African Names wrt Pleasant and Unpleasant  | European Names and African Names wrt Pleasant 2 and Unpleasant 2  | WEAT: Gender and ethnicity Queries average score  |
+========================+====================================================+===================================================+================================================+===============================================================+===================================================================+===================================================+
| glove twitter dim=25   | 0.715369                                           | 0.766402                                          | 0.121468                                       | NaN                                                           | 1.160488                                                          | 0.690932                                          |
+------------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| glove twitter dim=50   | 0.799666                                           | -0.660553                                         | -0.589894                                      | NaN                                                           | 1.007753                                                          | 0.764467                                          |
+------------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| glove twitter dim=100  | 0.681933                                           | 0.641153                                          | -0.399822                                      | NaN                                                           | 1.128199                                                          | 0.712777                                          |
+------------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+


Note that in the 4th column, all values are NaN. This is due the fact that the word set loaded lost more than 20% of the words when transformed into embeddings. 
This parameter is configurable.
On the other hand, note that the last column brings the average of the scores obtained per row. These are calculated from the absolute values of the values shown. 
In general, the Nan are Ignored. Like the previous parameter, these are also configurable.

Calculate Rankings
==================

If we run the same tests on another metric such as RNSB, we'll see that this one delivers results on a different scale than WEAT.

>>> from wefe.metrics import RNSB
>>> RNSB_queries_results = run_queries(RNSB,
>>>                       queries,
>>>                       models,
>>>                       queries_set_name='Gender and ethnicity Queries')

+-----------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| Model Name - Query    | Male terms and Female terms wrt Carrer and Family  | Male terms and Female terms wrt Science and Arts  | Male terms and Female terms wrt Math and Arts  | European Names and African Names wrt Pleasant and Unpleasant  | European Names and African Names wrt Pleasant 2 and Unpleasant 2  | RNSB: Gender and ethnicity Queries average score  |
+=======================+====================================================+===================================================+================================================+===============================================================+===================================================================+===================================================+
| glove twitter dim=25  | 0.003582                                           | 0.003099                                          | 0.045298                                       | NaN                                                           | 0.033330                                                          | 0.021327                                          |
+-----------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| glove twitter dim=50  | 0.021572                                           | 0.008006                                          | 0.056258                                       | NaN                                                           | 0.049533                                                          | 0.033842                                          |
+-----------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+
| glove twitter dim=100 | 0.008817                                           | 0.004689                                          | 0.061267                                       | NaN                                                           | 0.111471                                                          | 0.046561                                          |
+-----------------------+----------------------------------------------------+---------------------------------------------------+------------------------------------------------+---------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------+


A good solution is to make a ranking that allows you to compare scores on different scales. 

For this, we will use the create_ranking function.
It takes both results DataFrames and uses the average columns to create the rankings.

>>> from wefe.utils import create_ranking
>>> 
>>> ranking = create_ranking([WEAT_queries_results, RNSB_queries_results])
>>> ranking

+-----------------------+---------------------------------------------------+---------------------------------------------------+
| Model Name - Ranking  | WEAT: Gender and ethnicity Queries average score  | RNSB: Gender and ethnicity Queries average score  |
+=======================+===================================================+===================================================+
| glove twitter dim=25  | 3                                                 | 3                                                 |
+-----------------------+---------------------------------------------------+---------------------------------------------------+
| glove twitter dim=50  | 2                                                 | 1                                                 |
+-----------------------+---------------------------------------------------+---------------------------------------------------+
| glove twitter dim=100 | 1                                                 | 2                                                 |
+-----------------------+---------------------------------------------------+---------------------------------------------------+
