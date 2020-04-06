.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WEFE documentation!
============================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   create_metric
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples and Replications

   replications
   rank


`Getting started <quick_start.html>`_
-------------------------------------

Information regarding how to install and use WEFE.

`User Guide <user_guide.html>`_
-------------------------------

A guide from the most basic to the most complex about how to use the package. 
It is guided through code and contains several examples that can then be used to create your own experiments.

`Replication of paper experiments <replications.html>`_
-------------------------------------------------------

Replication of several results of experiments performed on bias measurement papers in Word Embeddings Among these are the experiments performed in the papers
   
   Semantics derived automatically from language corpora contain human-like biases.

And 
   A transparent framework for evaluating unintended demographic bias in word embeddings.


`API Documentation <api.html>`_
-------------------------------

It contains the list and specification of the functions and classes available in the package.


`Create your own Metric <create_metric.html>`_
----------------------------------------------

A complete guide on how to implement your own metrics using WEFE's interfaces and design.