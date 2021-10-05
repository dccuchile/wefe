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
Several studies has shown that they are prone to inherit stereotypical social
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

We originally proposed WEFE as a theoretical framework that aims to formalize the
main building blocks for measuring bias in word embedding models.
The purpose of developing this framework was to run a case study that consistently 
compares and ranks different embedding models.
Seeing the possibility that other research teams are facing the same problem, 
we decided to improve this code and publish it as a library, hoping that it 
can be useful for their studies.

Later, we realized that the library had the potential to cover more areas than just
bias measurement. This is why WEFE is constantly being improved, which so far has
resulted in a new bias mitigation module and multiple enhancements and fixes.

The main objectives we want to achieve with this library are:

- To provide a ready-to-use tool that allows the user to run bias tests in a 
  straightforward manner. 
- To provide a ready-to-use tool that allows the user to mitigate bias by means of a 
- simple fit-transform interface.
- To provide simple interface and utils to develop new metrics and mitigation methods.


Similar Packages
================

There are quite a few alternatives that complement WEFE. Be sure to check them out!

- Fair Embedding Engine: https://github.com/FEE-Fair-Embedding-Engine/FEE
- ResponsiblyAI: https://github.com/ResponsiblyAI/responsibly


Measurement Framework
=====================

Here we present the main building blocks of the measuring framework and then, we present 
the common usage pattern of WEFE. 

Target set 
----------

A target word set (denoted by :math:`T`) corresponds to a 
set of words intended to denote a particular social group,which is defined by a 
certain criterion. This criterion can be any character, trait or origin that 
distinguishes groups of people from each other e.g., gender, social class, age, 
and ethnicity. For example, if the criterion is gender we can use it to 
distinguish two groups, `women and men`. Then, a set of target words 
representing the social group "*women*" could contain words like “*she*”, 
“*woman*”, “*girl*”, etc. Analogously  a set of target words the representing the 
social group "*men*" could include “*he*”, “*man*”, “*boy*”, etc.


Attribute set
-------------

An attribute word set (denoted by :math:`A`) is a set of words 
representing some attitude, characteristic, trait, occupational field, etc.  
that  can  be  associated  with individuals from any social group. For example,
the set of *science* attribute  words  could  contain  words  such as  
*“technology”*, *“physics”*, *“chemistry”*, while the *art* attribute words could have
words like *“poetry”*,  *“dance”*,  *“literature”*.

Query
-----

Queries are the main building blocks used by fairness metrics to measure bias 
of word embedding models. 
Formally, a query is a pair :math:`Q=(\mathcal{T},\mathcal{A})` in which 
:math:`T` is a set of target word sets, and :math:`A` is a set of attribute 
word sets. For example, consider the target word sets:


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

.. math::

   \begin{equation}
   Q=(\{T_{\text{women}}, T_{\text{men}}\},\{A_{\text{science}},A_{\text{art}}\}).
   \end{equation}

When a set of queries :math:`\mathcal{Q} = {Q_1, Q_2, \dots, Q_n}` is intended
to measure a single type of bias, we say that the set has a 
**Bias Criterion**.  
Examples of bias criteria are gender, ethnicity, religion, politics, 
social class, among others.

.. warning::

  To accurately study the biases contained in word embeddings, queries may 
  contain words that could be offensive to certain groups or individuals. 
  The relationships studied between these words DO NOT represent the ideas, 
  thoughts or beliefs of the authors of this library.  
  This applies to this and all pages of the documentation. 



Query Template
--------------

A query template is simply a pair :math:`(t,a)\in\mathbb{N}\times\mathbb{N}`.
We say that query :math:`Q=(\mathcal{T},\mathcal{A})` satisfies a 
template :math:`(t,a)` if :math:`|\mathcal{T}|=t` and :math:`|\mathcal{A}|=a`.


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
Given an embedding model :math:`\mathbf{M}` and a query :math:`Q` that 
satisfies :math:`s_F`, the metric produces the value 
:math:`F(\mathbf{M},Q)\in \mathbb{R}` that quantifies the degree of bias of 
:math:`\mathbf{M}` with respect to query :math:`Q`.

Standard usage pattern of WEFE
-------------------------------

The following flow chart shows how to perform a bias measurement using a gender
query, word2vec embeddings and the WEAT metric.

.. image:: images/diagram_1.png
  :alt: Gender query with WEAT Flow

To see the implementation of this query using WEFE, refer to 
the `Quick start <quick_start.html>`_ section.



Metrics
