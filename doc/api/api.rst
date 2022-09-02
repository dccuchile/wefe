========
WEFE API
========

This reference details all the utilities as well as the metrics and mitigation methods
implemented so far in WEFE.

.. currentmodule:: wefe

WordEmbeddingModel
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.word_embedding_model.WordEmbeddingModel

Query
=====

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.query.Query


.. _metrics-API:

Metrics
=======

This list contains the metrics implemented in WEFE.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.metrics.WEAT
   wefe.metrics.RND
   wefe.metrics.RNSB
   wefe.metrics.MAC
   wefe.metrics.ECT
   wefe.metrics.RIPA

.. _debias-API:

Debias
======

This list contains the debiasing methods implemented so far in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.debias.HardDebias
   wefe.debias.MulticlassHardDebias
   wefe.debias.RepulsionAttractionNeutralization
   wefe.debias.DoubleHardDebias
   wefe.debias.HalfSiblingRegression

.. _datasets-API:

Datasets
===========

The following functions allow you to load sets of words used in previous studies.


.. autosummary::
   :toctree: generated/dataset/
   :template: function.rst

   wefe.datasets.load_bingliu
   wefe.datasets.fetch_debias_multiclass
   wefe.datasets.fetch_debiaswe
   wefe.datasets.fetch_eds
   wefe.datasets.load_weat


Preprocessing
=============


The following functions allow transforming sets of words and queries to embeddings. 
The documentation of the functions in this section are intended as a guide for WEFE developers.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   wefe.preprocessing.preprocess_word
   wefe.preprocessing.get_embeddings_from_set
   wefe.preprocessing.get_embeddings_from_tuples
   wefe.preprocessing.get_embeddings_from_query


Utils
=====

Collection of assorted utils. 

.. autosummary::
   :toctree: generated/
   :template: function.rst

   wefe.utils.load_test_model
   wefe.utils.generate_subqueries_from_queries_list
   wefe.utils.run_queries
   wefe.utils.plot_queries_results
   wefe.utils.create_ranking
   wefe.utils.plot_ranking
   wefe.utils.calculate_ranking_correlations
   wefe.utils.plot_ranking_correlations
   wefe.utils.flair_to_gensim
