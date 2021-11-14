===========
Quick Start
===========

In this tutorial we show you how to install WEFE and then how to run a 
basic query.


Download and setup
==================

There are two different ways to install WEFE: 

- To install the package with pip, run in a console::

    pip install --upgrade wefe

- To install the package with conda, run in a console::

    conda install -c pbadilla wefe 


Run your first Query
====================

.. warning::

  If you are not familiar with the concepts of query, target and attribute 
  set, please visit the `the framework section <about.html#the-framework>`_ 
  on the library's about page. 
  These concepts be widely used in the following sections.


In the following code we show how to implement the example query presented 
in WEFE's home page: A gender Query using WEAT metrics on the glove-twitter Word 
Embedding model. 

The following graphic shows the flow of the query execution:

.. image:: images/diagram_1.png
  :alt: Gender query with WEAT Flow

The programming of the previous flow can be separated into three steps:

- Load the Word Embedding model. 
- Create the Query. 
- Run the Query using the WEAT metric over the Word Embedding Model.

These stages be implemented next:

1. Load the Word Embedding pretrained model from :code:`gensim` and then, 
create a :code:`` instance with it.
This object took a gensim's :code:`KeyedVectors` object and a model name as 
parameters.
As we said previously, for this example, we use :code:`glove-twitter-25'` embedding model.

>>> # import the modules
>>> from wefe.query import Query
>>> from wefe.word_embedding_model import WordEmbeddingModel
>>> from wefe.metrics.WEAT import WEAT
>>> import gensim.downloader as api
>>>
>>> # load glove 
>>> twitter_25 = api.load('glove-twitter-25')
>>> model = WordEmbeddingModel(twitter_25, 'glove twitter dim=25')

2. Create the Query with a loaded, fetched or custom target and attribute 
word sets. In this case, we manually set both target words and attribute
words.

>>> # create the word sets
>>> target_sets = [['she', 'woman', 'girl'], ['he', 'man', 'boy']]
>>> target_sets_names = ['Female Terms', 'Male Terms']
>>>
>>> attribute_sets = [['poetry','dance','literature'], ['math', 'physics', 'chemistry']]
>>> attribute_sets_names = ['Arts', 'Science']
>>>
>>> # create the query
>>> query = Query(target_sets, attribute_sets, target_sets_names,
>>>               attribute_sets_names)

3. Instantiate the metric to be used and then, execute :code:`run_query` 
with the parameters created in the past steps. In this case we use the
`WEAT <about.html#weat>`_ metric. 

>>> # instance a WEAT metric
>>> weat = WEAT() 
>>> result = weat.run_query(query, model)
>>> print(result)
{
  'query_name': 'Female Terms and Male Terms wrt Arts and Science', 
  'result': 0.2595698336760204, 
  'weat': 0.2595698336760204, 
  'effect_size': 1.452482230821006, 
  'p_value': nan
}

A score greater than 0 indicates that there is indeed a biased relationship between 
women and the arts with respect to men and science. 

For more advanced usage, visit user the `User Guide <user_guide.html>`_ 
section.
