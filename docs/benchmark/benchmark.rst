Benchmark
=========

To the best of our knowledge, there are only three other libraries
besides WEFE that implement bias measurement and mitigation methods for
word embeddings: Fair Embedding Engine (FEE), Responsibly, and
EmbeddingBiasScores.

According to its authors, Fair Embedding Engine is defined as “A library
for analyzing and mitigating gender bias in word embeddings”,
Responsibly is defined as “A toolkit for auditing and mitigating bias
and fairness of machine learning systems”. Finally, EmbeddingBiasScores
describes itself as a collection of implementations and wrappers of bias
scores for text embeddings.

The documentation for these three libraries can be found at the
following links:

-  https://github.com/FEE-Fair-Embedding-Engine/FEE
-  https://docs.responsibly.ai/
-  https://github.com/HammerLabML/EmbeddingBiasScores

The benchmark presented here compares these three libraries against WEFE
according to the following criteria:

1. Ease of installation.
2. Quality of the package and documentation.
3. Ease of loading models.
4. Ease of running bias measurements.
5. Ease of running bias mitigation algorithms.
6. Implemented metrics and mitigation methods.

1. Ease of installation
-----------------------

This comparison aims to evaluate how easy it is to install the library.

WEFE
~~~~

According to the documentation, WEFE is available for installation using
the Python Package Index (via pip) as well as via conda.

.. code:: bash

   pip install --upgrade wefe
   # or
   conda install -c pbadilla wefe

Fair Embedding Engine
~~~~~~~~~~~~~~~~~~~~~

In the case of FEE, neither the documentation nor the repository
indicates how to install the package. Therefore, the easiest thing to do
in this case is to clone the repository and then install the
requirements , as described in the following steps:

1. Clone the repo

.. code:: bash

   $ git clone https://github.com/FEE-Fair-Embedding-Engine/FEE

2. Install the requirements.

.. code:: bash

   $ pip install -r FEE/requirements.txt
   $ pip install sympy
   $ pip install -U gensim==3.8.3

Responsibly
~~~~~~~~~~~

According to its documentation, responsibly is hosted in the Python
Package Index so it can be installed using pip.

.. code:: bash

   $ pip install responsibly

EmbeddingBiasScores
~~~~~~~~~~~~~~~~~~~

In the case of EmbeddingBiasScores, the documentation indicates that the
repository can be cloned and then installed locally.

.. code:: bash

   $ git clone https://github.com/HammerLabML/EmbeddingBiasScores.git
   $ pip install -r EmbeddingBiasScores/requirements.txt

Conclusion
~~~~~~~~~~

Both WEFE and Responsibly are hosted in the Python package index, which
simplifies their installation and dependency handling, lowering the
barrier to entry. FEE and EmbeddingBiasScores, on the other hand,
require ad hoc installation procedures that require more advanced
knowledge of Python and Pip.

2. Source Code Quality and Documentation
----------------------------------------

This benchmark seeks to compare the quality of documentation as well as
other software quality features such as testing and continuous
integration.

WEFE
~~~~

WEFE has a complete documentation site that explains in detail how to
use the package: an about page with the motivation and goals of the
project, a quick start page showing how to install the library, several
user guides on how to measure and mitigate bias in word embeddings, a
detailed API of the implemented methods, theoretical background in the
area, and finally implementations of previous case studies.

In addition, most of the code is tested and developed using continuous
integration mechanisms (through a linter and testing mechanisms in
Github Actions), which are well-established practices in software
development.

Fair Embedding Engine
~~~~~~~~~~~~~~~~~~~~~

FEE’ documentation covers only the basic aspects of the API and a
flowchart showing the main concepts of the library. The documentation
does not include user guides, code examples, or theoretical background
on the implemented methods.

In terms of software engineering practices and standards, no tests,
linter, or continuous integration mechanisms could be identified.

Responsibly
~~~~~~~~~~~

Responsibly has a complete documentation site that explains how to use
the package: an index page with the main project information and a quick
start page that shows how to install the library, demos that act as user
manuals, and a detailed API of the implemented methods.

In addition, most of the code is tested and developed using continuous
integration mechanisms (through a linter and testing in Github Actions).

EmbeddingBiasScores
~~~~~~~~~~~~~~~~~~~

It was not possible to find formal documentation explaining how to run
bias tests in EmbeddingBiasScores. There is only a small Jupyter
notebook with some use cases, which at the time of writing had several
flaws that made it difficult to understand and use.

No testing, linter, or continuous integration mechanisms could be
identified.

Conclusion
~~~~~~~~~~

In terms of documentation, WEFE contains much more detailed
documentation than the other libraries, with more extensive manuals and
replications of previous case studies. Responsibly has sufficient
documentation to execute its main functionalities without major
problems, however, it is not as exhaustive as that of WEFE. FEE, only
provides API documentation, which in our opinion is not sufficient for
new users to use it without problems. Finally, EmbeddingBiasScores only
presents a Jupyter notebook with some implementation examples.

With respect to software quality, both FEE and Responsibly comply with
well-established software development practices (i.e., testing,
continuous integration, linter). FEE and EmbeddingBiasScores, on the
other hand, do not have any of these practices in place

3. Ease of loading models
-------------------------

In this section we will compare how easy it is to load a pre-trained
word embedding (WE) model from each library. Two settings are compared:
loading a model from Gensim’s API (``glove-twitter-25``) and loading a
model from a binary file (``word2vec``).

The second setting requires downloading a WE model trained with the
original word2vec implementation, which can be obtained as follows:

.. code:: ipython3

    # !wget https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz
    # !gzip -dv word2vec-google-news-300.gz

WEFE
~~~~

In WEFE, WE models are represented internally by wrapping Gensim models.
This means that the model loading process (either from the API or from a
file) is handled by Gensim loaders, while the class that generates the
objects that allow access to the embeddings is managed by WEFE.

The following code shows how to load a glove model using the Gensim API
from within WEFE:

.. code:: ipython3

    from wefe.word_embedding_model import WordEmbeddingModel
    import gensim.downloader as api

    # load glove
    twitter_25 = api.load("glove-twitter-25")
    model = WordEmbeddingModel(twitter_25, "glove twitter dim=25")

The following code shows how to load a word2vec model trained with the
original implementation.

.. code:: ipython3

    from wefe.word_embedding_model import WordEmbeddingModel
    from gensim.models.keyedvectors import KeyedVectors

    # load word2vec
    word2vec = api.load("word2vec-google-news-300")
    model = WordEmbeddingModel(word2vec, "word2vec-google-news-300")

FEE
~~~

FEE also offers direct support for loading WE models from its API
through the following code. In this case, model loading is coupled to
the WE class, which provides the methods to access the embeddings.

.. code:: ipython3

    from FEE.fee.embedding.loader import WE

    fee_model = WE().load(ename="glove-twitter-25")

.. code:: ipython3

    from FEE.fee.embedding.loader import WE

    fee_model = WE().load(fname="word2vec-google-news-300", format="bin")

Responsibly and EmbeddingBiasScores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neither Responsibly nor EmbeddingBiasScores implement their own
interfaces to handle WE models. Users must rely on Gensim or other
external libraries for this purpose. This can be expressed as shown in
the following script:

.. code:: ipython3

    # load twitter_25 model from gensim api
    twitter_25 = api.load("glove-twitter-25")

    # load word2vec model from file
    word2vec = KeyedVectors.load_word2vec_format("word2vec-google-news-300", binary=True)

Conclusion
~~~~~~~~~~

As discussed above, both WEFE and FEE implement their own interfaces to
internally manage access to WE models. Responsibly and
EmbeddingBiasScores lack such functionalities, which may complicate
their use.

4. Ease of running bias measurements.
-------------------------------------

The following section aims to compare the execution of fairness metrics
in the libraries included in this study. To make the benchmark as
objective as possible, the set of words and the WE model are kept fixed
throughout the comparison, and only the metrics are allowed to vary.

.. code:: ipython3

    # words to evaluate

    female_terms = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    male_terms = ["male", "man", "boy", "brother", "he", "him", "his", "son"]

    family_terms = [
        "home",
        "parents",
        "children",
        "family",
        "cousins",
        "marriage",
        "wedding",
        "relatives",
    ]
    career_terms = [
        "executive",
        "management",
        "professional",
        "corporation",
        "salary",
        "office",
        "business",
        "career",
    ]

    # optional, only for wefe usage.
    target_sets_names = ["Female terms", "Male terms"]
    attribute_sets_names = ["Family terms", "Career terms"]

WEFE
~~~~

WEFE defines a standardized framework for executing metrics: in short,
it is necessary to define a query that will act as a container for the
words to be tested and then, together with the model, will be provided
as input to some metric.

The outputs of the metrics are contained in dictionaries that allow
additional metadata to be included to the output.

.. code:: ipython3

    # import the modules
    from wefe.query import Query

    # 1. create the query
    query = Query(
        [female_terms, male_terms],
        [family_terms, career_terms],
        target_sets_names,
        attribute_sets_names,
    )
    query




.. parsed-literal::

    <Query: Female terms and Male terms wrt Family terms and Career terms
    - Target sets: [['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter'], ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']]
    - Attribute sets:[['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives'], ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']]>



.. code:: ipython3

    from wefe.metrics.WEAT import WEAT

    # 2. instance a WEAT metric and pass the query plus the model.
    weat = WEAT()
    result = weat.run_query(query, model)
    result




.. parsed-literal::

    {'query_name': 'Female terms and Male terms wrt Family terms and Career terms',
     'result': 0.46343881433131173,
     'weat': 0.46343881433131173,
     'effect_size': 0.4507652792646716,
     'p_value': nan}



Since the ``run_query`` method is independent of the query and the
model, it can receive additional parameters that customize the process.
In this case, we show how to normalize the words before searching for
them in the model (i.e., lowercase them and remove their accents).

.. code:: ipython3

    weat = WEAT()
    result = weat.run_query(
        query,
        model,
        preprocessors=[{"lowercase": True, "strip_accents": True}],
    )
    result




.. parsed-literal::

    {'query_name': 'Female terms and Male terms wrt Family terms and Career terms',
     'result': 0.46343881433131173,
     'weat': 0.46343881433131173,
     'effect_size': 0.4507652792646716,
     'p_value': nan}



Next, we show how to report the corresponding p-value through a
permutation test.

.. code:: ipython3

    weat = WEAT()
    result = weat.run_query(
        query,
        model,
        calculate_p_value=True,
    )
    result




.. parsed-literal::

    {'query_name': 'Female terms and Male terms wrt Family terms and Career terms',
     'result': 0.46343881433131173,
     'weat': 0.46343881433131173,
     'effect_size': 0.4507652792646716,
     'p_value': 0.19068093190680932}



This interface allows us to easily switch to similar metrics (i.e.,
supporting the same number of number of word sets).

.. code:: ipython3

    from wefe.metrics import RNSB

    rnsb = RNSB()
    result = rnsb.run_query(query, model)
    result




.. parsed-literal::

    {'query_name': 'Female terms and Male terms wrt Family terms and Career terms',
     'result': 0.09051558681296493,
     'rnsb': 0.09051558681296493,
     'negative_sentiment_probabilities': {'female': 0.5285811053851917,
      'woman': 0.3031782770423851,
      'girl': 0.20810547466232254,
      'sister': 0.17327510211466302,
      'she': 0.4165425516161486,
      'her': 0.3895078245770702,
      'hers': 0.31412920848479164,
      'daughter': 0.13146512364633123,
      'male': 0.42679205714649815,
      'man': 0.43079499436045987,
      'boy': 0.21701323144255624,
      'brother': 0.19983034212661,
      'he': 0.5645185337599223,
      'him': 0.49470907399126185,
      'his': 0.552712793795697,
      'son': 0.17457869573293805},
     'negative_sentiment_distribution': {'female': 0.09565807331470504,
      'woman': 0.054866603359974946,
      'girl': 0.03766114329405169,
      'sister': 0.031357841309175544,
      'she': 0.07538229712572722,
      'her': 0.07048978417965314,
      'hers': 0.05684840897525258,
      'daughter': 0.02379143012863325,
      'male': 0.07723716469755836,
      'man': 0.0779615819300061,
      'boy': 0.03927319268906782,
      'brother': 0.036163580806998274,
      'he': 0.10216172076480977,
      'him': 0.0895282036894233,
      'his': 0.10002521923736822,
      'son': 0.03159375449759469}}



.. code:: ipython3

    from wefe.metrics import MAC

    mac = MAC()
    result = mac.run_query(query, model)
    result




.. parsed-literal::

    {'query_name': 'Female terms and Male terms wrt Family terms and Career terms',
     'result': 0.8416415235615204,
     'mac': 0.8416415235615204,
     'targets_eval': {'Female terms': {'female': {'Family terms': 0.9185737599618733,
        'Career terms': 0.916069650076679},
       'woman': {'Family terms': 0.752434104681015,
        'Career terms': 0.9377805145923048},
       'girl': {'Family terms': 0.707457959651947,
        'Career terms': 0.9867974997032434},
       'sister': {'Family terms': 0.5973392464220524,
        'Career terms': 0.9482253392925486},
       'she': {'Family terms': 0.7872791914269328,
        'Career terms': 0.9161583095556125},
       'her': {'Family terms': 0.7883057091385126,
        'Career terms': 0.9237247597193345},
       'hers': {'Family terms': 0.7385367527604103,
        'Career terms': 0.9480051446007565},
       'daughter': {'Family terms': 0.5472579970955849,
        'Career terms': 0.9277344475267455}},
      'Male terms': {'male': {'Family terms': 0.8735092766582966,
        'Career terms': 0.9468009045813233},
       'man': {'Family terms': 0.8249392118304968,
        'Career terms': 0.9350165261421353},
       'boy': {'Family terms': 0.7106057899072766,
        'Career terms': 0.9879048476286698},
       'brother': {'Family terms': 0.6280269809067249,
        'Career terms': 0.9477180293761194},
       'he': {'Family terms': 0.8693044614046812,
        'Career terms': 0.8771287016716087},
       'him': {'Family terms': 0.8230192996561527,
        'Career terms': 0.888683641096577},
       'his': {'Family terms': 0.8876195731572807,
        'Career terms': 0.8920885202242061},
       'son': {'Family terms': 0.5764635019004345,
        'Career terms': 0.9220191016211174}}}}



Fair Embedding Engine
~~~~~~~~~~~~~~~~~~~~~

In the case of Fair Embedding Engine, the WE model is passed in the
metric instantiation. Then, the output value of the metric is computed
using the ``compute`` method of the metric object.

FEE differs somewhat from the WEFE standardization by making mandatory
to provide the model when instantiating each metric, making the metric
object model dependent. This makes it difficult to test several models
at once since you have to instantiate a different metric object for each
model.

On the other hand, FEE does not establish a clear mechanism for passing
sets of words of different sizes to the computation method: sets of
words are delivered directly with a star parameter \*, which defines an
arbitrary number of positional arguments. This lack of definition makes
it difficult for the user to understand how many and which word sets to
pass.

.. code:: ipython3

    from FEE.fee.metrics import WEAT as FEE_WEAT

    fee_weat = FEE_WEAT(fee_model)

    fee_weat.compute(female_terms, male_terms, family_terms, career_terms)




.. parsed-literal::

    0.39821118



The FEE implementation of WEAT also allows the calculation of the
p-value.

.. code:: ipython3

    fee_weat.compute(female_terms, male_terms, family_terms, career_terms, p_val=True)




.. parsed-literal::

    (0.39821118, 0.0)



Finally, the implementation of the metric does not support the execution
of more complex actions, such as preprocessing word sets. We could not
find any other metric that was easily replaceable using the same or a
similar interface (with respect to the WEFE standardization layer).

Responsibly
~~~~~~~~~~~

Similar to WEFE, responsibly has a function that takes the model and
word sets as input and returns the WEAT score as output.

.. code:: ipython3

    from responsibly.we.weat import calc_single_weat

    calc_single_weat(
        twitter_25,
        first_target={"name": "female_terms", "words": female_terms},
        second_target={"name": "male_terms", "words": male_terms},
        first_attribute={"name": "family_terms", "words": family_terms},
        second_attribute={"name": "career_terms", "words": career_terms},
    )


.. parsed-literal::

    {'Target words': 'female_terms vs. male_terms',
     'Attrib. words': 'family_terms vs. career_terms',
     's': 0.31658393144607544,
     'd': 0.67794365,
     'p': 0.09673659673659674,
     'Nt': '8x2',
     'Na': '8x2'}



The p-value can also be obtained from the same function by setting the
``with_pvalue`` parameter to ``True``.

.. code:: ipython3

    calc_single_weat(
        twitter_25,
        first_target={"name": "female_terms", "words": female_terms},
        second_target={"name": "male_terms", "words": male_terms},
        first_attribute={"name": "family_terms", "words": family_terms},
        second_attribute={"name": "career_terms", "words": career_terms},
        with_pvalue=True,
    )




.. parsed-literal::

    {'Target words': 'female_terms vs. male_terms',
     'Attrib. words': 'family_terms vs. career_terms',
     's': 0.31658393144607544,
     'd': 0.67794365,
     'p': 0.09673659673659674,
     'Nt': '8x2',
     'Na': '8x2'}



The implementation of this metric does not include the ability to
perform more complex actions such as preprocessing word sets.

In addition, we were unable to find any metrics in this library other
than WEAT that are directly comparable to those implemented by WEFE.

EmbeddingBiasScores
~~~~~~~~~~~~~~~~~~~

EmbeddingBiasScores formalizes how bias is measured in a different way
than WEFE: it classifies the methods into clustering or geometric
methods (note that WEFE only implements the geometric equivalents).

As part of their standardization, each geometric metric must first
define the direction of the bias using the ``define_bias_space``
function with attribute_embeddings (attribute words) as input; and then
use the ``group_bias`` or ``mean_individual_bias`` methods to compute
the value of the metric.

Examples of use are shown below:

.. code:: ipython3

    # the embeddings to be used must be transformed by hand from words to arrays.
    target_embeddings = [
        [model[word] for word in female_terms],
        [model[word] for word in male_terms],
    ]
    attribute_embeddings = [
        [model[word] for word in family_terms],
        [model[word] for word in career_terms],
    ]

.. code:: ipython3

    from EmbeddingBiasScores.geometrical_bias import WEAT

    weat = WEAT()
    weat.define_bias_space(attribute_embeddings)
    # group bias returns the effect size.
    weat.group_bias(target_embeddings)




.. parsed-literal::

    0.4364516797305417



This implementation of WEAT returns the effect size by default. There is
no way to parameterize the metric to compute the WEAT score or the
p-value.

Similar to WEFE, the standardization implemented by EmbeddingBiasScores
allows to easily change the used metric to another with the same input
word sets.

.. code:: ipython3

    from EmbeddingBiasScores.geometrical_bias import MAC

    mac = MAC()
    mac.define_bias_space(attribute_embeddings)

    # mac does not accept more than one target set, so we have to calculate it manually.
    target_0_mac = mac.mean_individual_bias(target_embeddings[0])
    target_1_mac = mac.mean_individual_bias(target_embeddings[1])
    (target_0_mac + target_1_mac) / 2




.. parsed-literal::

    0.8416415235615204



EmbeddingBiasScores includes metrics that WEFE does not yet implement,
such as ``GeneralizedWEAT`` and ``SAME``.

.. code:: ipython3

    from EmbeddingBiasScores.geometrical_bias import GeneralizedWEAT

    gweat = GeneralizedWEAT()
    gweat.define_bias_space(attribute_embeddings)
    gweat.group_bias(target_embeddings)




.. parsed-literal::

    0.02896493



.. code:: ipython3

    from EmbeddingBiasScores.geometrical_bias import SAME

    same = SAME()
    same.define_bias_space(attribute_embeddings)
    same.mean_individual_bias(target_embeddings[0])




.. parsed-literal::

    0.2677120929221758



Finally, EmbeddingBiasScores does not allow any of its metrics to
perform more complex actions, such as preprocessing word set or
customizing some performance settings.

Conclusion
~~~~~~~~~~

In WEFE, having the input words as query objects decoupled from the
execution of metrics allows both parameterization of metric execution
and easy exchange of one metric for another. In addition, the clean and
unified interface for all metrics makes the execution of bias
measurements intuitive.

Responsibly and FEE share a similar interface, in which the metric
arguments are sets of words (which lack the expressiveness of WEFE
queries to declare the number of sets of words supported by each
metric), making it difficult to standardize inputs across metrics. We
were unable to find any metrics other than WEAT to include in the
benchmarking of FEE and Responsibly.

On the other hand, EmbeddingBiasScores also presents its own
mathematical standardization for each metric as well as some metrics
that WEFE does not yet implement. While the standardization they present
may be a bit more specific, it makes it more complex to use.

The increased difficulty is mainly due to two factors: users have to
manually define the bias space (using the ``define_bias_space``
parameter) and then investigate whether to use the parameters
``group_bias`` or ``mean_individual_bias``, which is not clear at first
sight unless the basics of the standardization proposed by this library
have been previously studied.

Finally, we highlight WEFE’s ``run_query`` method, which allows the user
to customize the execution of metrics, such as word preprocessing,
normalization of embeddings, and calculation of submetrics or
statistical tests.


5. Ease of Running Bias Mitigation Algorithms
---------------------------------------------

Next we will compare how to run bias mitigation methods on the libraries
included in the benchmark. In order to make the comparison as objective
as possible, the set of words and the embedding model remain fixed; only
the algorithms executed vary. Furthermore, to evaluate the performance
of the implemented methods, we will use the same query defined in the
previous section using WEAT (female vs. male terms with respect to
family vs. career).

.. code:: ipython3

    from wefe.datasets import fetch_debiaswe
    from wefe.utils import load_test_model

    # word sets to be used
    debiaswe_wordsets = fetch_debiaswe()

    definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    gender_specific = debiaswe_wordsets["gender_specific"]

    targets = [
        "executive",
        "management",
        "professional",
        "corporation",
        "salary",
        "office",
        "business",
        "career",
        "home",
        "parents",
        "children",
        "family",
        "cousins",
        "marriage",
        "wedding",
        "relatives",
    ]

WEFE
~~~~

WEFE defines a standardized framework for executing bias mitigation
algorithms based on the scikit-learn fit transform interface.

The fit-transform interface allows the user to select the sets of words
and parameters that will be used to learn the debiasing transformation
(``fit``), as well as to select the words that will be effectively
debiased by the method (``transform``).

This allows the user to change the words used to define the bias
criterion (which is usually gender, but could be easily changed), as
well as the vocabulary word to which the mitigation is applied. This
software design pattern is useful for comparing different de-biasing
methods, as the user can ensure that the same parameters are used across
methods.

Below we show how to execute a mitigation method with WEFE:

.. code:: ipython3

    from wefe.debias.hard_debias import HardDebias
    from wefe.debias.hard_debias import HardDebias
    from wefe.word_embedding_model import WordEmbeddingModel
    from gensim import downloader as api

    # load glove model
    twitter_25 = api.load("glove-twitter-25")
    model = WordEmbeddingModel(twitter_25, "glove twitter dim=25")

    # 1. instance Hard Debias algortihm
    hd = HardDebias(
        verbose=False,
        criterion_name="gender",
    )

    # 2. apply fit method and pass the model and definitional pairs.
    hd.fit(model, definitional_pairs=definitional_pairs)

    # 3. apply transform method passing the model, target and ignore word sets resulting in the debiased model
    hd_debiased_model = hd.transform(
        model,
        target=targets,
        ignore=gender_specific,
        copy=True,
    )


.. parsed-literal::

    Copy argument is True. Transform will attempt to create a copy of the original model. This may fail due to lack of memory.
    Model copy created successfully.


.. parsed-literal::

    100%|██████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 21809.84it/s]


Next, we show how to change the debiasing method while keeping a very
similar parameter configuration.

.. code:: ipython3

    from wefe.debias.repulsion_attraction_neutralization import (
        RepulsionAttractionNeutralization,
    )

    ran = RepulsionAttractionNeutralization().fit(
        model=model,
        definitional_pairs=definitional_pairs,
    )

    ran_debiased_model = ran.transform(
        model=model,
        target=targets,
        ignore=gender_specific,
        copy=True,
    )


.. parsed-literal::

    Copy argument is True. Transform will attempt to create a copyof the original model. This may fail due to lack of memory.
    Model copy created successfully.


.. parsed-literal::

    100%|█████████████████████████████████████████████████████████████████████████████| 16/16 [00:03<00:00,  5.23it/s]
    100%|██████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 45964.98it/s]


As can be seen, the fit-transform standardization implemented in WEFE
allows to easily execute and exchange the different bias mitigation
methods implemented in the library.

.. code:: ipython3

    from wefe.metrics import WEAT

    weat = WEAT()
    result = weat.run_query(
        query,
        model,
    )
    print("Original model WEAT evaluation: ", result["weat"])

    weat = WEAT()
    result = weat.run_query(
        query,
        hd_debiased_model,
    )
    print("Hard Debias debiased model WEAT evaluation: ", result["weat"])


    weat = WEAT()
    result = weat.run_query(
        query,
        ran_debiased_model,
    )
    print(
        "Repulsion Attraction Neutralization debiased model WEAT evaluation: ",
        result["weat"],
    )


.. parsed-literal::

    Original model WEAT evaluation:  0.31658415612764657
    Hard Debias debiased model WEAT evaluation:  0.002320525236427784
    Repulsion Attraction Neutralization debiased model WEAT evaluation:  0.26007230998948216


Fair Embedding Engine
~~~~~~~~~~~~~~~~~~~~~

The Fair Embedding Engine (FEE) requires the embedding model to be
passed during instantiation of the algorithm. It currently does not
support user-given definitional pairs, as the word sets used are fixed
in this implementation, focusing only on gender bias at the moment.

Debiasing is performed by executing the run method. The list of target
words to be debiased must be provided in this implementation.

.. code:: ipython3

    import copy
    from FEE.fee.embedding.loader import WE

    # load model
    fee_model = WE().load(ename="glove-twitter-25")
    # model must be normalized
    fee_model.normalize()

.. code:: ipython3

    from FEE.fee.debias import HardDebias

    # instance the algortihm and apply it to the embedding model
    fee_hd_debiased_model = HardDebias(copy.deepcopy(fee_model)).run(word_list=targets)

FEE allows easy use of different debiasing methods with a similar
interface

.. code:: ipython3

    from FEE.fee.debias import RANDebias

    # instance the algortihm and apply it to the embedding model
    ran_hd_debiased_model = RANDebias(copy.deepcopy(fee_model)).run(words=targets)


.. code:: ipython3

    # in the case, we generate a custom weat calculation using the fee debiasing methods.
    result = WEAT()._calc_weat(
        [fee_model.v(word) for word in query.target_sets[0]],
        [fee_model.v(word) for word in query.target_sets[1]],
        [fee_model.v(word) for word in query.attribute_sets[0]],
        [fee_model.v(word) for word in query.attribute_sets[1]],
    )

    print("Original model WEAT evaluation: ", result)
    result = WEAT()._calc_weat(
        [fee_hd_debiased_model.v(word) for word in query.target_sets[0]],
        [fee_hd_debiased_model.v(word) for word in query.target_sets[1]],
        [fee_hd_debiased_model.v(word) for word in query.attribute_sets[0]],
        [fee_hd_debiased_model.v(word) for word in query.attribute_sets[1]],
    )
    print("Hard Debias debiased model WEAT evaluation: ", result)
    result = WEAT()._calc_weat(
        [ran_hd_debiased_model.v(word) for word in query.target_sets[0]],
        [ran_hd_debiased_model.v(word) for word in query.target_sets[1]],
        [ran_hd_debiased_model.v(word) for word in query.attribute_sets[0]],
        [ran_hd_debiased_model.v(word) for word in query.attribute_sets[1]],
    )
    print("Repulsion Attraction Neutralization debiased model WEAT evaluation: ", result)


.. parsed-literal::

    Original model WEAT evaluation:  0.31658416730351746
    Hard Debias debiased model WEAT evaluation:  -0.061893132515251637
    Repulsion Attraction Neutralization debiased model WEAT evaluation:  0.17548414319753647



Responsibly
~~~~~~~~~~~

In Responsibly the embedding model is provided during the instantiation
of the ``GenderBiasWe`` class. Definitional pairs cannot be provided by
the user, as the bias being mitigated is set specifically to gender
bias. To perform the debiasing process, one simply needs to execute the
``debias`` method.

However, it should be noted that the mitigation method cannot be run on
the benchmark model chosen, as it is not compatible with uncased models
such as ``twitter-25``.

.. code:: ipython3

    from responsibly.we import GenderBiasWE

    # does not work with twitter_25.
    gender_bias_we = GenderBiasWE(word2vec)  # instance the GenderBiasWE
    gender_bias_we.debias(neutral_words=targets)  # apply the debias

EmbeddingBiasScore
~~~~~~~~~~~~~~~~~~

The library does not implement mitigation methods, so it is not included
in this comparison.

Conclusion
~~~~~~~~~~

All three libraries offer a simple way to apply bias mitigation
algorithms in a similar way and all of them are able to mitigate bias in
the word embedding model by similar amounts, depending on the metric
used.

The main difference between them is that WEFE offers more flexibility to
users, allowing them to choose the bias criteria through the words used
to learn the transformation and the words that are mitigated. On the
other hand, FEE and Responsibly only work with gender bias because the
set of words is fixed by default.

Finally, WEFE includes more mitigation algorithms than the other two
frameworks.

6. Metrics and Mitigation Methods Implemented
---------------------------------------------

The following tables provide a comparison of the libraries included in
this benchmarking, with respect to the bias metrics and mitigation
methods they implement to date.

Fairness Metrics
~~~~~~~~~~~~~~~~
====================== ==================== ===== ===== =========== ========================
Metric                Implementable in WEFE WEFE  FEE   Responsibly EmbeddingBiasScores
====================== ==================== ===== ===== =========== ========================
WEAT                  ✔                     ✔     ✔     ✔           ✔
WEAT ES               ✔                     ✔     ✖     ✖           ✖
RNSB                  ✔                     ✔     ✖     ✖           ✖
RIPA                  ✔                     ✔     ✖     ✖           ✔
ECT                   ✔                     ✔     ✖     ✖           ✖
RND                   ✔                     ✔     ✖     ✖           ✖
MAC                   ✔                     ✔     ✖     ✖           ✔
Direct Bias           ✔                     ✖     ✔     ✔           ✔
SAME                  ✔                     ✖     ✖     ✖           ✔
Generalized WEAT      ✔                     ✖     ✖     ✖           ✔
IndirectBias          ✖                     ✖     ✖     ✔           ✖
GIPE                  ✖                     ✖     ✔     ✖           ✖
PMN                   ✖                     ✖     ✔     ✖           ✖
Proximity Bias        ✖                     ✖     ✔     ✖           ✖
====================== ==================== ===== ===== =========== ========================

Metrics marked as "✔" in the "Implementable in WEFE" column can be implemented directly within
the WEFE framework using word sets as input.
Metrics marked as "✖" require additional representations such as gender directions
or apply before/after transformations, and are therefore currently out of WEFE's scope.


Mitigation algorithms
~~~~~~~~~~~~~~~~~~~~~

======================= ==== === =========== ===================
Algorithm               WEFE FEE Responsibly EmbeddingBiasScores
======================= ==== === =========== ===================
Hard Debias             ✔    ✔   ✔           ✖
Double Hard Debias      ✔    ✖   ✖           ✖
Half Sibling Regression ✔    ✔   ✖           ✖
RAN                     ✔    ✔   ✖           ✖
Multiclass HD           ✔    ✖   ✖           ✖
======================= ==== === =========== ===================

Conclusion
----------

The following table summarizes the main differences between the
libraries analyzed in this benchmark study.

 ==================================================== ========================================= ========================================================== ========================================== ====================================
 Item                                                 WEFE                                      FEE                                                        Responsibly                                EmbeddingBiasScores
 ==================================================== ========================================= ========================================================== ========================================== ====================================
  Implemented   Metrics                                7                                         5                                                          3                                          6
  Implemented   Mitigation Algorithms                  5                                         3                                                          1                                          0
  Extensible                                           Easy                                      Easy                                                       Difficult,   not very modular.             Easy
  Well-defined   interface for metrics                 ✔                                         ✖                                                          ✖                                          ✔
  Well-defined   interface for mitigation algorithms   ✔                                         ✖                                                          ✖                                          ✖
  Lastest update                                       July 2025                                 October 2020                                               April 2021                                 April 2023
  Installation                                         Easy:   pip or conda                      No instructions. It can be installed from the repository   Only   with pip. Presents problems         Only   from the repository
  Documentation                                        Extensive   documentation with examples   Almost   no documentation                                  Limited documentation with some examples   No   documentation, only examples.
 ==================================================== ========================================= ========================================================== ========================================== ====================================
