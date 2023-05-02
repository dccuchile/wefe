
Differences between IJCAI version and Current version
=====================================================

An initial iteration of the present software was constructed to execute the experiments
in our previous IJCAI publication titled "WEFE: The word embeddings fairness
evaluation framework" authored by Badilla, P., Bravo-Marquez, F., & Pérez, J. (2020)
presented at the International Joint Conferences on Artificial Intelligence.

It is pertinent to note that the primary focus of the IJCAI publication was the
conceptual framework of evaluating bias rather than the software's development.
The main differences between the previous version and the current one are discussed
below.


The most noticeable change we can mention with respect to the IJCAI
version and the current version is the full implementation of a new
debiasing methods module. It includes 5 methods of debiasing:
``HardDebias``, ``MulticlassHardDebias``, ``DoubleHardDebias``,
``RepulsionAttractionNeutralization`` and ``HalfSiblingRegression``.

Regarding metrics: The original version of WEFE published in IJCAI
contained 4 metrics: ``WEAT``, ``WEAT-ES``, ``RND`` and ``RNSB``.
Currently and thanks to contributions, WEFE also implements ``MAC``,
``RIPA`` and ``ECT``.

Also, the original version contained very rudimentary ``Query`` and
``WordEmbeddingModel`` wrapper routines.

In the actual version, the wrappers are much more complete and allow
better interaction with the user and with WEFE’s internal APIs.

For example, the implementation of ``__repr__`` for ``Query`` and
``WordEmbeddingModel`` contain short descriptions of each object for the
user. We have also included a ``dict`` method in ``Query`` that allows
to transform a query into a dictionary and the ``update`` in
``WordEmbeddingModel`` that allows to update an embedding associated to
a word by a new one.

The ``preprocessing`` module has also been improved to cover a wider
range of operations (such as different preprocessing steps) that have
been modularized and generalized so that any metric or mitigation method
can use it.

The documentation has been significantly improved from the original
release. These improvements include the addition of new user guides,
conceptual guides explaining the theoretical framework, multi-language
tutorials, and detailed API documentation covering metrics and
mitigation methods, including theoretical details. It is also worth
noting that there have been notable improvements in both testing and
code quality compared to the original release.
