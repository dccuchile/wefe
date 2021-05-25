============
Contributing
============

There are many ways to contribute to the library: 

- Implementing new metrics. A relatively extensive guide can be found in the 
  section `Creating your own metrics <create_metric.html>`_.
- Create more examples and use cases.
- Help to improve the documentation.
- Create more tests.

All contributions are welcome!

Get the repository
==================


You can download the library by running the following command ::

    git clone https://github.com/dccuchile/wefe


To contribute, simply create a pull request.
Verify that your code is well documented, to implement unit tests and 
follows the PEP8 coding style.

Testing
=======

All unit tests are located in the wefe/test folder and are based on the 
``pytest`` framework. 
In order to run tests, you will first need to install 
``pytest`` and ``pytest-cov``::

    pip install -U pytest
    pip install pytest-cov

To run the tests, execute::

    pytest wefe

To check the coverage, run::

    py.test wefe --cov-report xml:cov.xml --cov wefe

And then::

    coverage report -m


Build the documentation
=======================

The documentation is created using sphinx. It can be found in the doc folder 
at the project's root folder.
The documentation includes the API description and some tutorials.
To compile the documentation, run the following commands::

    cd doc
    make html 
