.. -*- mode: rst -*-

|ReadTheDocs|_ |CircleCI|_ |Conda|_ |CondaLatestRelease|_ |CondaVersion|_


.. |ReadTheDocs| image:: https://readthedocs.org/projects/wefe/badge/?version=latest
.. _ReadTheDocs: https://wefe.readthedocs.io/en/latest/?badge=latest


.. |CircleCI| image:: https://circleci.com/gh/dccuchile/wefe.svg?style=shield 
.. _CircleCI: https://circleci.com/gh/dccuchile/wefe.svg?style=shield 


.. |Conda| image:: https://anaconda.org/pbadilla/wefe/badges/installer/conda.svg
.. _Conda: https://anaconda.org/pbadilla/wefe/badges/installer/conda.svg


.. |CondaLatestRelease| image:: https://anaconda.org/pbadilla/wefe/badges/latest_release_date.svg
.. _CondaLatestRelease: https://anaconda.org/pbadilla/wefe/badges/latest_release_date.svg


.. |CondaVersion| image:: https://anaconda.org/pbadilla/wefe/badges/version.svg
.. _CondaVersion: https://anaconda.org/pbadilla/wefe/badges/version.svg




WEFE: The Word Embedding Fairness Evaluation Framework
======================================================


Word Embedding Fairness Evaluation (WEFE) is an open source library for measuring bias in word embedding models. It generalizes many existing fairness metrics into a unified framework and provides a standard interface for:

* Encapsulating existing fairness metrics from previous work and designing new ones.
* Encapsulating the test words used by fairness metrics into standard objects called queries.
* Computing a fairness metric on a given pre-trained word embedding model using user-given queries.

It also provides more advanced features for:

* Running several queries on multiple embedding models and return a DataFrame with the results.
* Plotting those results on a barplot.
* Based on the above results, calculating a bias ranking for all embedding models. This allows the user to evaluate the fairness of the embedding models according to the bias criterion (defined by the query) and the metric used.
* Plotting the ranking on a barplot.
* Correlating the rankings. This allows the user to see how the rankings of the different metrics or evaluation criteria are correlated with respect to the bias presented by the models.


The official documentation can be found at this `link <https://wefe.readthedocs.io/>`_.


Installation
============

There are two different ways to install WEFE: 


To install the package with ``pip``   ::

    pip install wefe

- With conda: 

To install the package with ``conda``::

    conda install -c pbadilla wefe 


Requirements
------------

These package will be installed along with the package, in case these have not already been installed:

1. numpy
2. scikit-learn
3. scipy
4. pandas
5. gensim
6. plotly


Contributing
------------

You can download the code executing ::

    git clone https://github.com/dccuchile/wefe


To contribute, visit the `Contributing <https://wefe.readthedocs.io/en/latest/contribute.html>`_ section in the documentation.


Testing
-------

All unit tests are in the wefe/test folder. It uses pytest as a framework to run them. 
You can run all tests, first install pytest and pytest-cov::

    pip install -U pytest
    pip install pytest-cov

To run the test, execute::

    pytest wefe

To check the coverage, run::

    py.test wefe --cov-report xml:cov.xml --cov wefe

And then::

    coverage report -m


Build the documentation
-----------------------

The documentation is created using sphinx. It can be found in the doc folder at the root of the project.
Here, the API is described as well as quick start and use cases.
To compile the documentation, run it::

    cd doc
    make html 


Citation
=========


Please cite the following paper if using this package in an academic publication:

P. Badilla, F. Bravo-Marquez, and J. Pérez 
`WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
29th International Joint Conference on Artificial Intelligence and the 17th 
Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. <https://www.ijcai.org/Proceedings/2020/60>`_

Bibtex:

::

@inproceedings{ijcai2020-60,
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





Team
====

- Pablo Badilla
- `Felipe Bravo-Marquez <https://felipebravom.com/>`_.
- `Jorge Pérez <https://users.dcc.uchile.cl/~jperez/>`_.


