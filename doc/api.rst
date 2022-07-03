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
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.query.Query



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


Debias
======

This list contains the mitigation (debiasing) methods implemented so far in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wefe.debias.HardDebias
   wefe.debias.MulticlassHardDebias
   wefe.debias.RepulsionAttractionNeutralization


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
   wefe.preprocessing.get_embeddings_from_sets
   wefe.preprocessing.get_embeddings_from_query

