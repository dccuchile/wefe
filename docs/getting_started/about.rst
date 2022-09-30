=====
About
=====

*Word Embedding Fairness Evaluation* (WEFE) is an open source library that implements 
many fairness metrics and mitigation methods (debias) in a unified framework. 
It also provides a standard interface for designing new ones. 

The main goal of the library is to provide a ready-to-use tool that allows the 
user to run bias measures and mitigation methods in a straightforward manner 
through well-designed and documented interfaces.

In bias measurement, WEFE provides a standard interface for:

- Encapsulating existing fairness metrics.
- Encapsulating the test words used by fairness metrics into standard
  objects called queries.
- Computing a fairness metric on a given pre-trained word embedding model 
  using user-given queries.


On the other hand, WEFE standardizes all mitigation methods through an interface 
inherited from `scikit-learn <https://scikit-learn.org/>`_ basic data transformations: 
the ``fit-transform`` interface. This standardization separates the mitigation 
process into two stages:

- The first step, ``fit``, learn the corresponding mitigation transformation.
- The ``transform`` method applies the transformation learned in the previous step
  to words residing in the original embedding space. 

.. note::

  To learn more about the measurement or mitigation framework, visit
  :ref:`measurement framework` or
  :ref:`mitigation framework` respectively, in the Conceptual Guides Section.

  For practical tutorials on how to measure or mitigate bias, visit 
  :ref:`bias measurement` or :ref:`bias mitigation` respectively 
  in the WEFE User Guide.

Motivation and objectives
=========================

Word Embedding models are a core component in almost all NLP downstream systems.
Several studies have shown that they are prone to inherit stereotypical social
biases from the corpus they were built on.
The common method for quantifying bias is to use a metric that calculates the
relationship between sets of word embeddings representing different social
groups and attributes.

Although previous studies have begun to measure bias in embeddings, they are
limited both in the types of bias measured (gender, ethnic) and in the models
tested. 
Moreover, each study proposes its own metric, which makes the relationship
between the results obtained unclear.

This fact led us to consider that we could use these metrics and studies to
make a case study in which we compare and rank the embedding models according
to their bias.

We originally proposed WEFE as a theoretical framework that formalizes the
main building blocks for measuring bias in word embedding models.
The purpose of developing this framework was to run a case study that consistently 
compares and ranks different embedding models.
Seeing the possibility that other research teams are facing the same problem, 
we decided to improve this code and publish it as a library, hoping that it 
can be useful for their studies.

We later realized that the library had the potential to cover more areas than just
bias measurement. This is why WEFE is constantly being improved, which so far has
resulted in a new bias mitigation module and multiple enhancements and fixes.

The main objectives we want to achieve with this library are:

- To provide a ready-to-use tool that allows the user to run bias tests in a 
  straightforward manner. 
- To provide a ready-to-use tool that allows the user to mitigate bias by means of a 
  simple `fit-transform` interface.
- To provide simple interface and utils to develop new metrics and mitigation methods.


Similar Packages
================

There are quite a few alternatives that complement WEFE. Be sure to check them out!

- Fair Embedding Engine: https://github.com/FEE-Fair-Embedding-Engine/FEE
- ResponsiblyAI: https://github.com/ResponsiblyAI/responsibly


Citation
=========

Please cite the following paper if using this package in an academic publication:

`P. Badilla, F. Bravo-Marquez, and J. Pérez WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the 29th International Joint Conference on Artificial Intelligence and the 17th Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. <https://www.ijcai.org/Proceedings/2020/60>`__

The author's version can be found at the following 
`link <https://felipebravom.com/publications/ijcai2020.pdf>`__.

Bibtex:

.. code-block:: latex 

    @InProceedings{wefe2020,
        title     = {WEFE: The Word Embeddings Fairness Evaluation Framework},
        author    = {Badilla, Pablo and Bravo-Marquez, Felipe and Pérez, Jorge},
        booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
                   Artificial Intelligence, {IJCAI-20}},
        publisher = {International Joint Conferences on Artificial Intelligence Organization},             
        pages     = {430--436},
        year      = {2020},
        month     = {7},
        doi       = {10.24963/ijcai.2020/60},
        url       = {https://doi.org/10.24963/ijcai.2020/60},
        }


Roadmap
=======

We expect in the future to:

- Implement measurement framework for contextualized embedding models.
- Implement new queries on different criteria.
- Create a single script that evaluates different embedding models under different bias criteria. 
- From the previous script, rank as many embeddings available on the web as possible.
- Implement a simple visualization module.
- Implement p-values mixin that applies for all metrics that accept two targets.

License
=======

WEFE is licensed under the BSD 3-Clause License.

Details of the license on this 
`link <https://github.com/dccuchile/wefe/blob/master/LICENSE>`__.

Team
====

- `Pablo Badilla <https://github.com/pbadillatorrealba/>`_.
- `Felipe Bravo-Marquez <https://felipebravom.com/>`_.
- `Jorge Pérez <https://users.dcc.uchile.cl/~jperez/>`_.
- `María José Zambrano  <https://github.com/mzambrano1/>`_.

Contributors
------------

We thank all our contributors who have allowed WEFE to grow, especially 
`stolenpyjak <https://github.com/stolenpyjak/>`_ and 
`mspl13 <https://github.com/mspl13/>`_ for implementing new metrics.

We also thank `alan-cueva <https://github.com/alan-cueva/>`_ for initiating the development 
of metrics for contextualized embedding models and 
`harshvr15 <https://github.com/harshvr15/>`_ for the examples of multi-language bias measurement.

Thank you very much 😊!

Contact
-------

Please write to pablo.badilla at ug.chile.cl for inquiries about the software. 
You are also welcome to do a pull request or publish an issue in the 
`WEFE repository on Github <https://github.com/dccuchile/wefe/>`_.

Acknowledgments
===============

This work was funded by the 
`Millennium Institute for Foundational Research on Data (IMFD) <https://imfd.cl/en/>`_.
It is also sponsored by `National Center of Artificial Intelligence of Chile (CENIA) <https://cenia.cl/en/>`_.