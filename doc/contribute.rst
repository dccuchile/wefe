============
Contributing
============

There are several tasks and features you can contribute to: 
- Implementing new metrics. A relatively extensive guide can be in `Create your own Metric <create_metric.html>`_ section.
- Create more examples and use cases
- Improve documentation
- Create more tests
- Refactoring the code

Among many others. All contributions are welcome. 
We'll be very happy for you to make them!!

Get the repository
==================


You can download the code executing ::

    git clone https://github.com/pabloBad/wefe.git


To contribute, simply create a pull request.
Verify that your code has documentation, test and format (PEP8)


Testing
=======

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
=======================

The documentation is created using sphinx. It can be found in the doc folder at the root of the project.
Here, the API is described as well as quick start and use cases.
To compile the documentation, run it::

    cd doc
    make html 
