========
WEFE API
========

This is the documentation of the API of WEFE. 

.. currentmodule:: wefe

WordEmbeddingModel
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WordEmbeddingModel

Query
=====

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Query


.. _metrics-API:

Metrics
=======

This list contains the metrics implemented in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WEAT
   RND
   RNSB
   MAC
   ECT
   RIPA


Debias
======

This list contains the debiasing methods implemented so far in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HardDebias
   MulticlassHardDebias
   DoubleHardDebias
   HalfSiblingRegression


Dataloaders
===========

The following functions allow one to load word sets used in previous works. 


.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   load_bingliu
   fetch_debias_multiclass
   fetch_debiaswe
   fetch_eds
   load_weat

Preprocessing
=============


The following functions allow transforming sets of words and queries to embeddings. 
The documentation of the functions in this section are intended as a guide for WEFE developers.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocess_word
   get_embeddings_from_set
   get_embeddings_from_tuples
   get_embeddings_from_query