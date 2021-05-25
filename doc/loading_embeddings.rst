=========================================
Loading embeddings from different sources
=========================================

WEFE depends on gensim's :code:`KeyedVectors` to operate the word 
embeddings models.
Therefore, any embedding you want to experiment with must be a model loaded 
through gensim's APIs or any library that extends it.

In technical terms, the minimum requirement for WEFE to operate with a model
is that it extends the :code:`BaseKeyedVectors` class.

Next we show several options to load models using different sources.

Create a example query
======================

In this section we only create an example query (same as the query of user guide)
to be used in the following sections.


>>> # Load the query
>>> from wefe.query import Query
>>> from wefe.word_embedding import 
>>> from wefe.metrics.WEAT import WEAT
>>> from wefe.datasets.datasets import load_weat
>>> 
>>> # load the weat word sets
>>> word_sets = load_weat()
>>> 
>>> # create the query
>>> query = Query([word_sets['male_terms'], word_sets['female_terms']],
>>>               [word_sets['career'], word_sets['family']],
>>>               ['Male terms', 'Female terms'], 
>>>               ['Career', 'Family'])
>>>
>>> # instantiate the metric
>>> weat = WEAT()

Load from Gensim API
====================

Gensim provides an 
`extensive list of pre-trained models <https://github.com/RaRe-Technologies/gensim-data#models>`_ 
that can be used directly. Below we show an example of use.

>>> import gensim.downloader as api
>>> 
>>> # Load from gensim.downloader some model, for example: glove-twitter-25
>>> glove_25_keyed_vectors = api.load('glove-twitter-25')
>>> 
>>> # The resulting object is already a BaseKeyedVectors subclass object.
>>> # so we can wrap directly using .
>>> glove_25_model = (glove_25_keyed_vectors, 'glove-25')
>>> 
>>> # Execute the query
>>> result = weat.run_query(query, glove_25_model)
>>> print(result)
{'query_name': 'Male terms and Female terms wrt Career and Family', 'result': 0.33814692}


Using Gensim Load
=================

As we said before, any model that is loaded with gensim and extends
:code:`BaseKeyedVectors` can be used in WEFE to measure bias.
In this section we will see how to load a word2vec model and Fasttext.

.. note::
  Gensim is not directly compatible with glove model file format. 
  However, they provide a 
  `script <https://radimrehurek.com/gensim/scripts/glove2word2vec.html>`_
  that allows you to transform any glove model into a word2vec format.


Loading Word2vec
----------------

For example, let's load word2vec from a .bin file
The procedure is quite simple: first we download word2vec binary file from its source
and then we load it using the :code:`KeyedVectors.load_word2vec_format` function.

>>> from gensim.models import KeyedVectors
>>> 
>>> w2v_embeddings = KeyedVectors.load_word2vec_format("/path/to/your/embeddings/model", binary=True)
>>> word2vec = (w2v_embeddings, 'word2vec')
>>> 
>>> result = weat.run_query(query, word2vec)
>>> result
{'query_name': 'Male terms and Female terms wrt Career and Family',
 'result': 0.7280304}


Loading FastText
----------------

The same method works for :code:`Fasttext`.

>>> from gensim.models import KeyedVectors
>>> fast_embeddings = KeyedVectors.load_word2vec_format('path/to/fast/embeddings.vec')
>>> 
>>> fast = (fast_embeddings, 'fast')
>>> result = weat.run_query(query, fast)
>>> 
>>> result
{'query_name': 'Male terms and Female terms wrt Career and Family',
 'result': 0.34870023}

While we load FastText here as :code:`KeyedVectors` (i.e. in word2vec format), 
it can also be used via :code:`FastTextKeyedVectors`.


Flair
=====

WEFE does not yet support flair interfaces.
However, you can use static embeddings of flair 
(
`Classic Word Embeddings <https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md>`_ 
) which are based on gensim's :code:`KeyedVectors`, to load embedding models.
The following code is an example of this:

>>> from flair.embeddings import WordEmbeddings
>>> 
>>> glove_embedding = WordEmbeddings('glove') # 100 dim glove
>>> 
>>> # extract KeyedVectors object
>>> glove_keyed_vectors = glove_embedding.precomputed_word_embeddings 
>>> glove_100 = (glove_keyed_vectors, 'glove-100')
>>> 
>>> result = weat.run_query(query, glove_100)
>>> print(result)
{'query_name': 'Male terms and Female terms wrt Career and Family', 'result': 1.0486683}