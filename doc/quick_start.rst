#####################
Quick Start with WEFE
#####################


WEFE is a package focused on providing an easy and well-designed framework for measuring word embedding bias. 

It provides metrics, a framework for creating queries, and a standard interface for executing these queries using a metric and a pre-trained Word Embedding model.
In addition, it has multiple tools that allow you to run several queries on several different embedding models.

Although it is only in its early stages of development, it is expected that with time it will become more robust, that more metrics will be implemented and that it will extend to other types of bias measurement in NLP.


The following lines will show the quick start documentation for WEFE.
In this page, we will show how to install the package and how to run a basic query.

1. Download and setup
=====================

There are two different ways to install WEFE: 

To install the package with pip, run in a console::

    pip install wefe

To install the package with conda, run in a console::

    conda install wefe



2. Run your first Query
=======================


The following code will show how to run aquery using one Word Embedding and a particular metric (WEAT).
The common flow to perform a query in WEFE consist in three steps, which will be displayed next to the code:

>>> from wefe.query import Query
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics.WEAT import WEAT
>>> import gensim.downloader as api

1. Load the Word Embedding pretrained model from gensim and then, create a WordEmbeddingModel instance with it.
For this example, we will use a twitter_25 .

>>> twitter_25 = api.load('glove-twitter-25')
>>> model = WordEmbeddingModel(twitter_25, 'glove twitter dim=25')

2. Create the Query with a loaded, fetched or custom target and attribute word sets.
For this example, we will create a query with gender terms with respect to arts and science.

>>> target_sets = [['male', 'man', 'boy'], ['female', 'woman', 'girl']]
>>> target_sets_names = ['Male Terms', 'Female Terms']
>>>
>>> attribute_sets = [['poetry', 'art', 'dance'], ['science','technology','physics']]
>>> attribute_sets_names = ['Arts', 'Science']
>>>
>>> query = Query(target_sets, attribute_sets, target_sets_names,
>>>               attribute_sets_names)

3. Instance the metric that you will use and then, execute run_query with the parameters created in the past steps. In this case we will use WEAT. 

>>> weat = WEAT()
>>> result = weat.run_query(query, model)
>>> print(result)
{'query_name': 'Male Terms and Female Terms wrt Arts and Science',
 'result': -0.010003209}

For more advanced examples, visit user `User Guide <user_guide.html>`_ section.