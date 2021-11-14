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


To install the package with ```pip```   ::

    pip install wefe

- With conda: 

To install the package with ```conda```::

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

Changelog
=========

Version 0.3.0
-------------
- Implemented Bolukbasi et al. 2016 Hard Debias.
- Implemented  Thomas Manzini et al. 2019 Multiclass Hard Debias.
- Implemented a fetch function to retrieve gn-glove female-male word sets.
- Moved the transformation logic of words, sets and queries to embeddings to its own module: preprocessing
- Enhanced the preprocessor_args and secondary_preprocessor_args metric preprocessing parameters to an list of preprocessors `preprocessors` together with the parameter `strategy` indicating whether to consider all the transformed words (`'all'`) or only the first one encountered (`'first'`).
- Renamed WordEmbeddingModel attributes ```model``` and ```model_name```  to ```wv``` and ```name``` respectively.
- Renamed every run_query ```word_embedding``` argument to ```model``` in every metric.


Version 0.2.2
-------------

- Added RIPA metrics (thanks @stolenpyjak for your contribution!).
- Fixed Literal typing bug to make WEFE compatible with python 3.7.

Version 0.2.1
-------------

- Compatibility fixes.


Version 0.2.0
--------------

- Renamed optional ```run_query``` parameter  ```warn_filtered_words``` to `warn_not_found_words`.
- Added ```word_preprocessor_args``` parameter to ```run_query``` that allow specifying transformations prior to searching for words in word embeddings.
- Added ```secondary_preprocessor_args``` parameter to ```run_query``` which allows specifying a second pre-processor transformation to words before searching them in word embeddings. It is not necessary to specify the first preprocessor to use this one.
- Implemented ```__getitem__``` function in ```WordEmbeddingModel```. This method allows obtaining an embedding from a word from the model stored in the instance using indexers. 
- Removed underscore from class and instance variable names.
- Improved type and verification exception messages when creating objects and executing methods.
- Fix an error that appeared when calculating rankings with two columns of aggregations with the same name.
- Ranking correlations are now calculated using pandas ```corr``` method. 
- Changed metric template, name and short_names to class variables.
- Implemented ```random_state``` in RNSB to allow replication of the experiments.
- run_query now returns as a result the default metric requested in the parameters and all calculated values that may be useful in the other variables of the dictionary.
- Fixed problem with api documentation: now it shows methods of the classes.
- Implemented p-value for WEAT


Citation
=========


Please cite the following paper if using this package in an academic publication:

P. Badilla, F. Bravo-Marquez, and J. Pérez 
`WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
29th International Joint Conference on Artificial Intelligence and the 17th 
Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. <https://www.ijcai.org/Proceedings/2020/60>`_

Bibtex:
::

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


Team
====

- Pablo Badilla
- `Felipe Bravo-Marquez <https://felipebravom.com/>`_.
- `Jorge Pérez <https://users.dcc.uchile.cl/~jperez/>`_.


