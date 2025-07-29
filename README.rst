.. -*- mode: rst -*-

|License|_ |GithubActions|_ |ReadTheDocs|_ |Downloads|_ |Pypy|_ |CondaVersion|_

.. |License| image:: https://img.shields.io/github/license/dccuchile/wefe
.. _License: https://github.com/dccuchile/wefe/blob/master/LICENSE

.. |ReadTheDocs| image:: https://readthedocs.org/projects/wefe/badge/?version=latest
.. _ReadTheDocs: https://wefe.readthedocs.io/en/latest/?badge=latest

.. |GithubActions| image:: https://github.com/dccuchile/wefe/actions/workflows/ci.yaml/badge.svg?branch=master
.. _GithubActions: https://github.com/dccuchile/wefe/actions

.. |Downloads| image:: https://pepy.tech/badge/wefe
.. _Downloads: https://pepy.tech/project/wefe

.. |Pypy| image:: https://badge.fury.io/py/wefe.svg
.. _Pypy: https://pypi.org/project/wefe/

.. |CondaVersion| image:: https://anaconda.org/pbadilla/wefe/badges/version.svg
.. _CondaVersion: https://anaconda.org/pbadilla/wefe


WEFE: The Word Embedding Fairness Evaluation Framework
======================================================

.. image:: ./docs/logos/WEFE_2.png
  :width: 300
  :alt: WEFE Logo
  :align: center

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


The official documentation can be found at this `link <https://wefe.readthedocs.io/>`_.


Installation
============

WEFE requires Python 3.10 or higher. There are two different ways to install WEFE:

**Install with pip** (recommended)::

    pip install wefe

**Install with conda**::

    conda install -c pbadilla wefe

**Install development version**::

    pip install git+https://github.com/dccuchile/wefe.git

**Install with development dependencies**::

    pip install "wefe[dev]"

**Install with PyTorch support**::

    pip install "wefe[pytorch]"


Requirements
------------

WEFE automatically installs the following dependencies:

- gensim (>=3.8.3)
- numpy (<=1.26.4)
- pandas (>=2.0.0)
- plotly (>=6.0.0)
- requests (>=2.22.0)
- scikit-learn (>=1.5.0)
- scipy (<1.13)
- semantic_version (>=2.8.0)
- tqdm (>=4.0.0)

Contributing
------------

To contribute to WEFE development:

1. **Clone the repository**::

    git clone https://github.com/dccuchile/wefe
    cd wefe

2. **Install in development mode with all dependencies**::

    pip install -e ".[dev]"

3. **Run tests to ensure everything works**::

    pytest tests

4. **Make your changes and run tests again**

5. **Follow our coding standards**:
   - Use ``ruff`` for code formatting: ``ruff format .``
   - Check code quality: ``ruff check .``
   - Run type checking: ``mypy wefe``

For detailed contributing guidelines, visit the `Contributing <https://wefe.readthedocs.io/en/latest/user_guide/contribute.html>`_ section in the documentation.

Development Requirements
------------------------

To install WEFE with all development dependencies for testing, documentation building, and code quality tools::

    pip install "wefe[dev]"

This installs additional packages including:

- pytest and pytest-cov for testing
- sphinx and related packages for documentation
- ruff for code formatting and linting
- mypy for type checking
- ipython for interactive development


Testing
-------

All unit tests are in the ``tests/`` folder. WEFE uses ``pytest`` as the testing framework.

To run all tests::

    pytest tests

To run tests with coverage reporting::

    pytest tests --cov=wefe --cov-report=html

To run a specific test file::

    pytest tests/test_datasets.py

Coverage reports will be generated in ``htmlcov/`` directory.


Build the documentation
-----------------------

The documentation is built using Sphinx and can be found in the ``docs/`` folder.

To build the documentation::

    cd docs
    make html

Or using the development environment::

    pip install "wefe[dev]"
    cd docs
    make html

The built documentation will be available at ``docs/_build/html/index.html``

Changelog
=========

Version 1.0.0
-------------------

**Major Release - Breaking Changes**

- **Python 3.10+ Required**: Dropped support for Python 3.6-3.9
- **Modern Packaging**: Migrated from ``setup.py`` to ``pyproject.toml``
- **Updated Dependencies**: All packages updated for modern Python ecosystem

**New Features**:

- Robust dataset fetching with retry mechanism and exponential backoff
- HTTP 429 (rate limiting) and timeout error handling
- Optional dependencies: ``pip install "wefe[dev]"`` and ``"wefe[pytorch]"``
- Dynamic version loading from ``wefe.__version__``

**Core Improvements**:

- **WordEmbeddingModel**: Enhanced type safety, better gensim compatibility, improved error handling
- **BaseMetric**: Refactored input validation, standardized ``run_query`` methods across all metrics
- **Testing**: Converted to pytest patterns with monkeypatch, comprehensive test coverage
- **Code Quality**: Migration from flake8 to Ruff, enhanced documentation with detailed docstrings

**Development Workflow**:

- GitHub Actions upgraded with Python 3.10-3.13 matrix testing
- Pre-commit hooks enhanced with JSON/TOML validation and security checks
- Modernized Sphinx documentation configuration
- Updated benchmark documentation and metrics comparison tables

Version 0.4.1
-------------------

- Fixed a bug where the last pair of target words in RIPA was not included.
- Added a benchmark that compares WEFE with another measurement and bias mitigation
  libraries in the documentation.
- Added a library changes since original paper release page in the documentation.

Version 0.4.0
-------------------
- 3 new bias mitigation methods (debias) implemented: Double Hard Debias, Half
  Sibling Regression and Repulsion Attraction Neutralization.
- The library documentation of the library has been restructured.
  Now, the documentation is divided into user guide and theoretical framework
  The user guide does not contain theoretical information.
  Instead, theoretical documentation can be found in the conceptual guides.
- Improved API documentation and examples. Added multilingual examples contributed
  by the community.
- The user guides are fully executable because they are now on notebooks.
- There was also an important improvement in the API documentation and in metrics and
  debias examples.
- Improved library testing mechanisms for metrics and debias methods.
- Fixed wrong repr of query. Now the sets are in the correct order.
- Implemented repr for WordEmbeddingModel.
- Testing CI moved from CircleCI to GithubActions.
- License changed to MIT.

Version 0.3.2
-------------
- Fixed RNSB bug where the classification labels were interchanged and could produce
  erroneous results when the attributes are of different sizes.
- Fixed RNSB replication notebook
- Update of WEFE case study scores.
- Improved documentation examples for WEAT, RNSB, RIPA.
- Holdout parameter added to RNSB, which allows to indicate whether or not a holdout
  is performed when training the classifier.
- Improved printing of the RNSB evaluation.

Version 0.3.1
-------------
- Update WEFE original case study
- Hotfix: Several bug fixes for execute WEFE original Case Study.
- fetch_eds top_n_race_occupations argument set to 10.
- Preprocessing: get_embeddings_from_set now returns a list with the lost
  preprocessed words instead of the original ones.

Version 0.3.0
-------------
- Implemented Bolukbasi et al. 2016 Hard Debias.
- Implemented  Thomas Manzini et al. 2019 Multiclass Hard Debias.
- Implemented a fetch function to retrieve gn-glove female-male word sets.
- Moved the transformation logic of words, sets and queries to embeddings to its own
  module: preprocessing
- Enhanced the preprocessor_args and secondary_preprocessor_args metric
  preprocessing parameters to an list of preprocessors `preprocessors` together with
  the parameter `strategy` indicating whether to consider all the transformed words
  (`'all'`) or only the first one encountered (`'first'`).
- Renamed WordEmbeddingModel attributes ```model``` and ```model_name```  to
  ```wv``` and ```name``` respectively.
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

- Renamed optional ```run_query``` parameter  ```warn_filtered_words``` to
  `warn_not_found_words`.
- Added ```word_preprocessor_args``` parameter to ```run_query``` that allow specifying
  transformations prior to searching for words in word embeddings.
- Added ```secondary_preprocessor_args``` parameter to ```run_query``` which allows
  specifying a second pre-processor transformation to words before searching them in
  word embeddings. It is not necessary to specify the first preprocessor to use this
  one.
- Implemented ```__getitem__``` function in ```WordEmbeddingModel```. This method
  allows obtaining an embedding from a word from the model stored in the instance
  using indexers.
- Removed underscore from class and instance variable names.
- Improved type and verification exception messages when creating objects and executing
  methods.
- Fix an error that appeared when calculating rankings with two columns of aggregations
  with the same name.
- Ranking correlations are now calculated using pandas ```corr``` method.
- Changed metric template, name and short_names to class variables.
- Implemented ```random_state``` in RNSB to allow replication of the experiments.
- run_query now returns as a result the default metric requested in the parameters
  and all calculated values that may be useful in the other variables of the dictionary.
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
