.. -*- mode: rst -*-

.. |Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. .. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. .. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. .. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. .. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. .. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. .. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. .. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. .. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. .. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-template/badge/?version=latest
.. .. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

#######################################################
WEFE: The Word Embeddings Fairness Evaluation Framework
#######################################################


WEFE is a package focused on providing an easy and well-designed framework for measuring word embedding bias. 

It provides metrics, a framework for creating queries, and a standard interface for executing these queries using a metric and a pre-trained Word Embedding model.
In addition, it has multiple tools that allow you to run several queries on several different embedding models, graph them, calculate their associated rankings per test, among others.

Although it is only in its early stages of development, it is expected that with time it will become more robust, that more metrics will be implemented and that it will extend to other types of bias measurement in NLP.

The official documentation can be found at: LINK TODO


Requirements
------------

These package will be installed along with the package, in case these have not already been installed:

1. numpy
2. scikit-learn
3. scipy
4. pandas
5. gensim
6. plotly
7. patool


Installation
============

There are two different ways to install WEFE: 


- With pip: 

To install the package with pip, run in a console:

    pip install wefe

- With conda: 

To install the package with conda, run in a console: 

    conda install wefe


Requirements
------------

These package will be installed along with the package, in case these have not already been installed:

1. numpy
2. scikit-learn
3. scipy
4. pandas
5. gensim
6. plotly
7. patool


Development
===========


Contributing
------------

You can download the code executing 

    git clone https://github.com/pabloBad/wefe.git


To contribute, simply create a pull request.
Verify that your code has documentation, test and format (PEP8)


Testing
-------

All unit tests are in the wefe/test folder. It uses pytest as a framework to run them. 
You can run all tests, first install pytest and pytest-cov:

    pip install -U pytest
    pip install pytest-cov

To run the test, execute:

    pytest wefe

To check the coverage, run:

    py.test wefe --cov-report xml:cov.xml --cov wefe

And then: 

    coverage report -m


Build the documentation
-----------------------

The documentation is created using sphinx. It can be found in the doc folder at the root of the project.
Here, the API is described as well as quick start and use cases.
To compile the documentation, run it: 

    cd doc
    make html 

