=====
About
=====

*Word Embedding Fairness Evaluation* (WEFE) is an open source library for 
measuring an mitigating bias in word embedding models. 
It generalizes many existing fairness metrics into a unified framework and 
provides a standard interface for:

- Encapsulating existing fairness metrics from previous work and designing
  new ones.
- Encapsulating the test words used by fairness metrics into standard
  objects called queries.
- Computing a fairness metric on a given pre-trained word embedding model 
  using user-given queries.

WEFE also standardizes the process of mitigating bias through an interface similar 
to the ``scikit-learn`` ``fit-transform``.
This standardization separates the mitigation process into two stages:

- The logic of calculating the transformation to be performed on the model (``fit``).
- The execution of the mitigation transformation on the model (``transform``).

Motivation and objectives
=========================

Word Embeddings models are a core component in almost all NLP downstream systems.
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

`P. Badilla, F. Bravo-Marquez, and J. Pérez 
 WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
 29th International Joint Conference on Artificial Intelligence and the 17th 
 Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. <https://www.ijcai.org/Proceedings/2020/60>`_

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

- Implement the metrics that have come out in recent works on bias in embeddings.
- Implement new queries on different criteria.
- Create a single script that evaluates different embedding models under different bias criteria. 
- From the previous script, rank as many embeddings available on the web as possible.
- Implement a visualization module.
- Implement p-values with statistic resampling to all metrics.

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