.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WEFE documentation!
==============================

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
   contribute
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   replications
   rank

About
=====

Word Embedding Fairness Evaluation Framework (WEFE) is a package focused on providing an easy and well-designed framework for measuring word embedding bias. 

It provides metrics, a framework for creating queries, and a standard interface for executing these queries using a metric and a pre-trained Word Embedding model.
In addition, it has multiple tools that allow you to run several queries on several different embedding models, graph them, calculate their associated rankings per test, among others.

Although it is only in its early stages of development, it is expected that with time it will become more robust, that more metrics will be implemented and that it will extend to other types of bias measurement in NLP.


Documentation
=============

The following pages contain information about how to install the package, how to use it and how to contribute, as well as detailed API documentation and extensive examples. 

`Quick start with WEFE <quick_start.html>`_
-------------------------------------------

Information regarding how to install and use WEFE.

`User Guide <user_guide.html>`_
-------------------------------

A guide from the most basic to the most complex about how to use the package. 
It is guided through code and contains several examples that can then be used to create your own experiments.

`Replication of paper experiments <replications.html>`_
-------------------------------------------------------

Replication of several results of experiments performed on bias measurement papers in Word Embeddings 
   

`API Documentation <api.html>`_
-------------------------------

It contains the list and specification of the functions and classes available in the package.


`Create your own Metric <create_metric.html>`_
----------------------------------------------

A complete guide on how to implement your own metrics using WEFE's interfaces and design.


`Contribute <contribute.html>`_
----------------------------------------------

A little guide on how to contribute to the project.


Relevant Papers
===============

A collection of papers related to WEFE. 
These include those that implement metrics and experiments as well as those related to the study of bias and bias reduction in Word Embeddings.

Measures and Experiments 
------------------------


- `Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186. <http://www.cs.bath.ac.uk/~jjb/ftp/CaliskanSemantics-Arxiv.pdf>`_.
- `Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. Proceedings of the National Academy of Sciences, 115(16), E3635-E3644. <https://www.pnas.org/content/pnas/115/16/E3635.full.pdf>`_.
- `Sweeney, C., & Najafian, M. (2019, July). A Transparent Framework for Evaluating Unintended Demographic Bias in Word Embeddings. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 1662-1667). <https://www.aclweb.org/anthology/P19-1162.pdf>`_.
- `Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings <https://arxiv.org/pdf/1904.04047>`_.


Bias Mitigation
---------------

- `Bolukbasi, T., Chang, K. W., Zou, J., Saligrama, V., & Kalai, A. (2016). Quantifying and reducing stereotypes in word embeddings. arXiv preprint arXiv:1606.06121. <https://arxiv.org/pdf/1606.06121.pdf>`_.
- `Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in neural information processing systems (pp. 4349-4357). <http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf>`_.
- `Zhao, J., Zhou, Y., Li, Z., Wang, W., & Chang, K. W. (2018). Learning gender-neutral word embeddings. arXiv preprint arXiv:1809.01496. <https://arxiv.org/pdf/1809.01496.pdf>`_.
- `Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2017). Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457. <https://arxiv.org/pdf/1707.09457.pdf>`_.
- `Gonen, H., & Goldberg, Y. (2019). Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. arXiv preprint arXiv:1903.03862. <https://arxiv.org/pdf/1903.03862.pdf>`_.

Surveys and other resources
___________________________


A Survey on Bias and Fairness in Machine Learning

- `Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2019). A survey on bias and fairness in machine learning. arXiv preprint arXiv:1908.09635. <https://arxiv.org/pdf/1908.09635.pdf>`_.
- `Bakarov, A. (2018). A survey of word embeddings evaluation methods. arXiv preprint arXiv:1801.09536. <https://arxiv.org/pdf/1801.09536.pdf>`_.
- `Camacho-Collados, J., & Pilehvar, M. T. (2018). From word to sense embeddings: A survey on vector representations of meaning. Journal of Artificial Intelligence Research, 63, 743-788. <https://www.jair.org/index.php/jair/article/view/11259/26454>`_.


Citations
=========

We do not have any paper published yet 😭😭😭😭😭


Team
====

- Pablo Badilla
- `Felipe Bravo Marquez <https://felipebravom.com/>`_.
- `Jorge Perez <https://users.dcc.uchile.cl/~jperez/>`_.
