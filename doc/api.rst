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
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Query


Metrics
==========

This list contains the metrics implemented in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WEAT


.. autosummary::
   :toctree: generated/
   :template: class.rst

   RND


.. autosummary::
   :toctree: generated/
   :template: class.rst

   RNSB


.. autosummary::
   :toctree: generated/
   :template: class.rst

   MAC


.. autosummary::
   :toctree: generated/
   :template: class.rst

   ECT


.. autosummary::
   :toctree: generated/
   :template: class.rst

   RIPA


Debias
======

This list contains the debiasing methods implemented so far in WEFE.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HardDebias

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MulticlassHardDebias

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HalfSiblingRegression


Dataloaders
===========

The following functions allow one to load word sets used in previous works. 


.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   load_bingliu


.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   fetch_debias_multiclass



.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   fetch_debiaswe


.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   fetch_eds


.. autosummary::
   :toctree: generated/dataloaders/
   :template: function.rst

   load_weat

Preprocessing
=============


The following functions allow transforming sets of words and queries to embeddings. 
The documentation of the functions in this section are intended as a guide for WEFE developers.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocess_word


.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_embeddings_from_set

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_embeddings_from_sets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_embeddings_from_query