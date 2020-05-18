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

Word Embedding Fairness Evaluation (WEFE) is an open source library for measuring bias in word embedding models. It generalizes many existing fairness metrics into a unified framework and provides a standard interface for:

- Encapsulating existing fairness metrics from previous work and design new ones.
- Encapsulating the test words used by fairness metrics into standard objects called queries.
- Computing a fairness metric on a given pre-trained word embedding model using user-given queries.


It also provides more advanced features for:

- Running several queries on multiple embedding models and return a 
  DataFrame with the results.
- Plotting those results on a barplot.
- Based on the above results, calculating a bias ranking for all embedding models. 
  This allows the user to evaluate the fairness of the embedding models according to
  the bias criterion (defined by the query) and the metric used.
- Plotting the ranking on a barplot.
- Correlating the rankings. This allows the user to see how the rankings of the different metrics or evaluation criteria are correlated with respect to the bias presented by the models.
 
  
  

Motivation and objectives
=========================

The measurement of bias in word embedding models have existed for some time. 
The common approach is to compute a metric based on the relationship between the embeddings of different word sets,
where the words from these sets represent social groups and general attributes of people.

Each of these metrics was specifically designed for the study in which they were proposed. This leads to a lack of consistency between them, which causes several problems when trying to compare and validate their results.

In order to address the above, our framework is based on the following objectives:

- To provide a ready-to-use tool that allows the user to run bias tests in a straightforward manner. 
- To provide simple interface to develop new metrics.
- To solve the two main problems that arise when comparing experiments based on different metrics:
   - Some metrics operate with different number of word sets as input. 
   - The outputs of different metrics are incompatible with each other 
     (their scales are different, some metrics return real numbers and 
     others only positive ones, etc..)


The Framework
=============

We will now present the basic concepts for the operation of the framework. 
Then, we will present the different possible flows. 

Target set 
----------

A target word set (denoted by :math:`T`) corresponds to a 
set of words intended to denote a particular social group,which is defined by a 
certain criterion. This criterion can be any character, trait or origin that 
distinguishes groups of people from each other e.g., gender, social class, age, 
and ethnicity. For example, if the criterion is gender we can use it to 
distinguish two groups, `women and men`. Then,  a set of target words 
representing the women social group could con-tain  words  like  “she”, 
“woman”, “girl”, etc. Analogously,the target words for the men social group 
could include “he”, “man”, “boy”, etc. I


Attribute set
-------------

An attribute word set (denoted by :math:`A`) is a set of words 
representing some attitude, characteristic, trait, occupational field, etc.  
that  can  be  associated  with individuals from any social group. For example,
the set of science attribute  words  could  contain  words  such as  
“technology”, “physics”, “chemistry”, while the art attribute words could have
words like “poetry”,  “dance”,  “literature”.

Query
-----

Queries are the main building blocks used by fairness metrics to measure bias 
of word embedding models. 
Formally, a query is a pair :math:`Q=(\mathcal{T},\mathcal{A})` in which :math:`T` is a set
of target word sets, and :math:`A` is a set of attribute word sets.
For example, consider the target word sets:


.. math::

   \begin{eqnarray*}
   T_{\text{women}} & = & \{{she},{woman},{girl}, \ldots\}, \\
   T_{\text{men}} & = & \{{he},{man},{boy}, \ldots\},
   \end{eqnarray*}

and the attribute word sets

.. math::

   \begin{eqnarray*}
   A_{\text{science}} & = & \{{math},{physics},{chemistry}, \ldots\}, \\
   A_{\text{art}} & = & \{{poetry},{dance},{literature}, \ldots\}.
   \end{eqnarray*}

Then the following is a query in our framework

Query Template
--------------

A query template is simply a pair :math:`(t,a)\in\mathbb{N}\times\mathbb{N}`.
We say that query :math:`Q=(\mathcal{T},\mathcal{A})` satisfies a template :math:`(t,a)` if 
:math:`|\mathcal{T}|=t` and :math:`|\mathcal{A}|=a`.

Fairness Measure
----------------

A fairness metric is a function that quantifies the degree of association 
between target and attribute words in a word embedding model. 
In our framework, every fairness metric is defined as a function that has a 
query and a model as input, and produces a real number as output.

Several fairness metrics have been proposed in the literature.
But not all of them share a common input template for queries.
Thus, we assume that every fairness metric comes with a template that 
essentially defines the shape of the input queries supported by the metric. 

Formally, let :math:`F` be a fairness metric with template :math:`s_F=(t_F,a_F)`. 
Given an embedding model :math:`\mathbf{M}` and a query :math:`Q` that satisfies :math:`s_F`, 
the metric produces the value :math:`F(\mathbf{M},Q)\in \mathbb{R}` that
quantifies the degree of bias of :math:`\mathbf{M}` with respect to query :math:`Q`.

Flow of a Bias measure on a Embedding model
-------------------------------------------

The following flow chart shows how to perform a bias measurement using a gender
query, word2vec and WEAT metric.

.. image:: images/diagram_1.png
  :alt: Fair RNSB sentiment distribution


Metrics
=======

Although it is only in its early stages of development, it is expected that 
with time it will become more robust, that more metrics will be implemented 
and that it will extend to other types of bias measurement in NLP.

The metrics implemented in the package so far are:

WEAT
----

Word Embedding Association Test (WEAT), presented in the paper Semantics 
derived automatically from language corpora contain human-like biases.
This metric receives two sets :math:`T_1` and :math:`T_2` of target words, 
and two sets :math:`A_1` and :math:`A_2` of attribute words. Its objective is 
to quantify the strength of association of both pair of sets through a 
permutation test. 
It also conaints a vairant, WEAT Effect Size. This variant represents a 
normalized measure that quantifies how far apart the two distributions of 
association between targets and attributes are.

RND
---

Relative Norm Distance (RND), presented in the paper Word embeddings quantify 
100 years of gender and ethnic stereotypes. RND averages the embeddings of 
each target set, then for each of the attribute words, calculates the norm 
of the difference between the word and the average target, and then subtracts 
the norms. The more positive (negative) the relative distance from the norm, 
the more associated are the sets of attributes towards group two (one). 

RNSB
----

Relative Negative Sentiment Bias (RNSB), presented in the paper A transparent 
framework for evaluating unintended demographic bias in word embeddings.

RNSB receives as input queries with two attribute sets :math:`A_1` and 
:math:`A_2` and two or more target sets, and thus has a template of the 
form :math:`s=(N,2)` with :math:`N\geq 2`.
Given a query :math:`Q=(\{T_1,T_2,\ldots,T_n\},\{A_1,A_2\})` and an embedding 
model :math:`\mathbf{M}`, in order to compute the metric :math:`F_{\text{RNSB}}(\mathbf{M},Q)` 
one first constructs a binary classifier :math:`C_{(A_1,A_2)}(\cdot)` using set 
:math:`A_1` as training examples for the negative class, and :math:`A_2` as 
training examples for the positive class. 
After the training process, this classifier gives for every word :math:`w` a 
probability :math:`C_{(A_1,A_2)}(w)` that can be interpreted as the degree of 
association of $w$ with respect to  :math:`A_2 (value $1-C_{(A_1,A_2)}(w)` is 
the degree of association with :math:`A_1`).
Now, we construct a probability distribution :math:`P(\cdot)` over all the words 
:math:`w` in :math:`T_1\cup \cdots \cup T_n`, by computing :math:`C_{(A_1,A_2)}(w)` 
and normalizing it to ensure that :math:`\sum_w P(w)=1`.
The main idea behind RNSB is that the more that :math:`P(\cdot)` resembles a 
uniform distribution, the less biased the word embedding model is.

MAC
---

Mean Average Cosine Similarity (MAC), presented in the paper Black is to 
criminalas caucasian is to police: Detecting and removing multiclass bias 
in word embeddings.  


Documentation
=============

The following pages contain information about how to install the package, 
how to use it and how to contribute, as well as detailed API documentation 
and extensive examples. 

`Quick start with WEFE <quick_start.html>`_
-------------------------------------------

Information regarding how to install and use WEFE.

`User Guide <user_guide.html>`_
-------------------------------

A guide from the most basic to the most complex about how to use the package. 
It is guided through code and contains several examples that can then be used 
to create your own experiments.

`Replication of paper experiments <replications.html>`_
-------------------------------------------------------

Replication of several results of experiments performed on bias measurement 
papers in Word Embeddings 
   

`API Documentation <api.html>`_
-------------------------------

It contains the list and specification of the functions and classes available 
in the package.


`Create your own Metric <create_metric.html>`_
----------------------------------------------

A complete guide on how to implement your own metrics using WEFE's interfaces 
and design.


`Contribute <contribute.html>`_
----------------------------------------------

A little guide on how to contribute to the project.


Relevant Papers
===============

A collection of papers related to WEFE. 
These include those that implement metrics and experiments as well as those 
related to the study of bias and bias reduction in Word Embeddings.

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


Citation
=========


Please cite the following paper if using this package in an academic publication:

P. Badilla, F. Bravo-Marquez, and J. Pérez 
WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
29th International Joint Conference on Artificial Intelligence and the 17th 
Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. 

Bibtex:
::
   @InProceedings{wefe2020,
     author    = {Pablo Badilla, Felipe Bravo-Marquez, and Jorge Pérez},
     title     = {WEFE: The Word Embeddings Fairness Evaluation Framework},
     booktitle = {Proceedings of the 29th International Joint Conference on Artificial Intelligence and the 17th Pacific Rim  International Conference on Artificial Intelligence (IJCAI-PRICAI 2020)},
     year      = {2020},
   }



Team
====

- Pablo Badilla
- `Felipe Bravo-Marquez <https://felipebravom.com/>`_.
- `Jorge Perez <https://users.dcc.uchile.cl/~jperez/>`_.
