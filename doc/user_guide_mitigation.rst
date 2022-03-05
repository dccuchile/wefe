Bias Mitigation
===============

WEFE also provides several methods to mitigate the bias of the embedding
models. In the following section:

*  We present how to reduce binary bias (such as gender bias) using Hard
   Debias.
*  We present how to reduce multiclass bias (such as ethnic having
   classes like black, white, Latino, etc…) using Multiclass Hard
   Debias.


.. code:: python

    import gensim.downloader as api
    
    from wefe.datasets import fetch_debiaswe, load_weat
    from wefe.debias.hard_debias import HardDebias
    from wefe.metrics import WEAT
    from wefe.query import Query
    from wefe.word_embedding_model import WordEmbeddingModel
    
    twitter_25 = api.load("glove-twitter-25")
    model = WordEmbeddingModel(twitter_25, "glove-twitter-dim=25")

Hard Debias
-----------

This method allow reducing the bias of an embedding model through
geometric operations between embeddings. This method is binary because
it only allows two classes of the same bias criterion, such as male or
female.

The main idea of this method is:

1. **Identify a bias subspace through the defining sets.** In the case
    of gender, these could be
    e.g., \ ``{'woman', 'man'}, {'she', 'he'}, ...``

2. **Neutralize the bias subspace on the embeddings that should not be
   biased.**

    First, it is defined a set of words that are correct to be related to
    the bias criterion: the *criterion specific gender words*. For
    example, in the case of gender, *gender specific* words are:
    ``{'he', 'his', 'He', 'her', 'she', 'him', 'him', 'She', 'man', 'women', 'men'...}``.

    Then, it is defined that all words outside this set should have no
    relation to the bias criterion and thus have the possibility of being
    biased. (e.g. for the case of gender: ``{doctor, nurse, ...}``).
    Therefore, this set of words is neutralized with respect to the bias
    subspace found in the previous step.

    The neutralization is carried out under the following operation:

    -  u : embedding
    -  v : bias direction

    First calculate the projection of the embedding on the bias subspace.

    -  projection = v • (v • u) / (v • v)

    Then subtract the projection from the embedding.

    -  u’ = u - projection

3. **Equalize the embeddings with respect to the bias direction.**.

    Given an equalization set (set of word pairs such as [she, he], [men,
    women], …, but not limited to the definitional set) this step
    executes, for each pair, an equalization with respect to the bias
    direction. That is, it takes a pair of embeddings and distributes
    them both at the same distance from the bias direction, so that
    neither is closer to the bias direction than the other.

Fit-Transform Interface
~~~~~~~~~~~~~~~~~~~~~~~

WEFE implements all debias methods through an interface inspired by the
transformers of ``scikit-learn``. That is, the execution of a debias
method involves two steps: - First a training through the ``fit`` method
where the transformation that will be applied on the embeddings is
calculated - Second, a ``transform`` that applies the trained
transformation.

Each of these stages defines its own parameters.

The fit parameters define how the neutralization will be calculated. In
Hard Debias, you have to provide the the ``definitional_pairs``, the
``equalize_pairs`` (which could be the same of definitional pairs) and
optionally, a debias ``criterion_name`` (to name the debiased model).

.. code:: python

    debiaswe_wordsets = fetch_debiaswe()
    
    definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    equalize_pairs = debiaswe_wordsets["equalize_pairs"]
    gender_specific = debiaswe_wordsets["gender_specific"]

    hd = HardDebias(verbose=False, criterion_name="gender").fit(
        model,
        definitional_pairs=definitional_pairs,
        equalize_pairs=equalize_pairs,
    )


The parameters of the transform method are relatively standard for all
methods. The most important ones are ``target``, ``ignore`` and
``copy``.

In the following example we use ``ignore`` and ``copy``, which are
described below:

-  ``ignore`` (by default, ``None``):

    A list of strings that indicates that the debias method will perform
    the debias in all words except those specified in this list. In case
    it is not specified, debias will be executed on all words. In case
    ignore is not specified or its value is None, the transformation will
    be performed on all embeddings. This may cause words that are
    specific to social groups to lose that component (for example,
    leaving ``'she'`` and ``'he'`` without a gender component).

-  ``copy`` (by default ``True``):

    if the value of copy is ``True``, method attempts to create a copy of
    the model and run debias on the copy. If ``False``, the method is
    applied on the original model, causing the vectors to mutate.

    **WARNING:** Setting copy with ``True`` requires at least 2x RAM of
    the size of the model. Otherwise the execution of the debias may raise
    ``MemoryError``.

Next, the transformation is executed using a copy of the model,
ignoring the words contained in ``gender_specific``.

.. code:: python

    gender_debiased_model = hd.transform(model, ignore=gender_specific, copy=True)


.. parsed-literal::

    Copy argument is True. Transform will attempt to create a copy of the original model. This may fail due to lack of memory.
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    Model copy created successfully.
    100%|██████████| 1193514/1193514 [00:18<00:00, 65143.18it/s]
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors


Using the metrics displayed in the first section of this user guide, we
can measure whether or not there was a change in the measured bias
between the original model and the debiased model.

.. code:: python

    weat_wordset = load_weat()
    weat = WEAT()
    
    gender_query_1 = Query(
        [word_sets["male_terms"], word_sets["female_terms"]],
        [word_sets["career"], word_sets["family"]],
        ["Male terms", "Female terms"],
        ["Career", "Family"],
    )
    
    gender_query_2 = Query(
        [weat_wordset["male_names"], weat_wordset["female_names"]],
        [weat_wordset["pleasant_5"], weat_wordset["unpleasant_5"]],
        ["Male Names", "Female Names"],
        ["Pleasant", "Unpleasant"],
    )

.. code:: python

    biased_results_1 = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results_1 = weat.run_query(gender_query, gender_debiased_model, normalize=True)
    
    print(round(debiased_results_1["weat"], 3),"<",round(biased_results_1["weat"], 3),
        "=",debiased_results_1["weat"] < biased_results_1["weat"],)

.. parsed-literal::

    -0.06 < 0.317 = True


.. code:: python

    biased_results_2 = weat.run_query(
        gender_query_2, model, normalize=True, preprocessors=[{}, {"lowercase": True}]
    )
    debiased_results_2 = weat.run_query(
        gender_query_2,
        gender_debiased_model,
        normalize=True,
        preprocessors=[{}, {"lowercase": True}],
    )
    
    print(
        round(debiased_results_2["weat"], 3),"<",round(biased_results_2["weat"], 3),
        "=",debiased_results_2["weat"] < biased_results_2["weat"],)

.. parsed-literal::

    -1.033 < -0.949 = True


Target Parameter
~~~~~~~~~~~~~~~~


-  target: If a set of words is specified in target, the debias method will be performed
   only on the word embeddings associated with this set. In the case of providing
   ``None``, the transformation will be performed on all vocabulary words except those
   specified in ignore. By default ``None``.

   In the following example, the target parameter is used to execute the transformation 
   only on the career and family word set:

.. code:: python

    targets = ['executive',
               'management',
               'professional',
               'corporation',
               'salary',
               'office',
               'business',
               'career',
               'home',
               'parents',
               'children',
               'family',
               'cousins',
               'marriage',
               'wedding',
               'relatives']

    hd = HardDebias(verbose=False, criterion_name="gender").fit(
        model,
        definitional_pairs=definitional_pairs,
        equalize_pairs=equalize_pairs,
    )
    
    gender_debiased_model = hd.transform(
        model, target=targets, copy=True
    )


.. parsed-literal::

    Copy argument is True. Transform will attempt to create a copy of the original model. This may fail due to lack of memory.
    Model copy created successfully.
    100%|██████████| 16/16 [00:00<00:00, 10754.63it/s]
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors


Next, a bias test is run on the mitigated embeddings associated with the
target words. In this case, the value of the metric is lower on the
query executed on the mitigated model than on the original one.
These results indicate that there was a mitigation of bias on embeddings of these words.

.. code:: python

    biased_results_1 = weat.run_query(gender_query_1, model, normalize=True)
    debiased_results_1 = weat.run_query(gender_query, gender_debiased_model, normalize=True)
    
    print(round(debiased_results_1["weat"], 3),"<",round(biased_results_1["weat"], 3)
          ,"=",debiased_results_1["weat"] < biased_results_1["weat"],)


.. parsed-literal::

    -0.06 < 0.317 = True


However, if a bias test is run with words that were outside the target
word set, the results are almost the same. The slight difference in the
metric scores lies in the fact that the equalize sets were still
equalized.
Equalization can be deactivated by delivering an empty equalize set (``[]``)

.. code:: python

    biased_results_2 = weat.run_query(
        gender_query_2, model, normalize=True, preprocessors=[{}, {"lowercase": True}]
    )
    debiased_results_2 = weat.run_query(
        gender_query_2,
        gender_debiased_model,
        normalize=True,
        preprocessors=[{}, {"lowercase": True}],
    )
    
    print(round(debiased_results_2["weat"], 3),"<",round(biased_results_2["weat"], 3),
        "=",debiased_results_2["weat"] < biased_results_2["weat"],)


.. parsed-literal::

    -0.941 < -0.949 = False


Save the Debiased Model
~~~~~~~~~~~~~~~~~~~~~~~

To save the mitigated model one must access the ``KeyedVectors`` (the
gensim object that contains the embeddings) through ``wv`` and then use
the ``save`` method to store the method in a file.

.. code:: python

    gender_debiased_model.wv.save('gender_debiased_glove.kv')


.. parsed-literal::

    INFO:gensim.utils:saving Word2VecKeyedVectors object under gender_debiased_glove.kv, separately None
    INFO:gensim.utils:storing np array 'vectors' to gender_debiased_glove.kv.vectors.npy
    INFO:gensim.utils:not storing attribute vectors_norm
    DEBUG:smart_open.smart_open_lib:{'uri': 'gender_debiased_glove.kv', 'mode': 'wb', 'buffering': -1, 'encoding': None, 'errors': None, 'newline': None, 'closefd': True, 'opener': None, 'ignore_ext': False, 'compression': None, 'transport_params': None}
    INFO:gensim.utils:saved gender_debiased_glove.kv


Multiclass Hard Debias
----------------------

Multiclass Hard Debias is a generalized version of Hard Debias that
enables multiclass debiasing. Generalized refers to the fact that this
method extends Hard Debias in order to support more than two types of
social target sets within the definitional set.

For example, for the case of religion bias, it supports a debias using
words associated with Christianity, Islam and Judaism.

The usage is very similar to Hard Debias with the difference that the
``definitional_sets`` can be larger than pairs.

.. code:: python

    from wefe.datasets import fetch_debias_multiclass
    from wefe.debias.multiclass_hard_debias import MulticlassHardDebias
    
    multiclass_debias_wordsets = fetch_debias_multiclass()
    weat_wordsets = load_weat()
    weat = WEAT()
    
    ethnicity_definitional_sets = multiclass_debias_wordsets["ethnicity_definitional_sets"]
    ethnicity_equalize_sets = list(
        multiclass_debias_wordsets["ethnicity_analogy_templates"].values()
    )
    
    mhd = MulticlassHardDebias(verbose=True, criterion_name="ethnicity")
    mhd.fit(
        model=model,
        definitional_sets=ethnicity_definitional_sets,
        equalize_sets=ethnicity_equalize_sets,
    )
    
    ethnicity_debiased_model = mhd.transform(model, copy=True)


.. parsed-literal::

    INFO:wefe.debias.multiclass_hard_debias:PCA variance explaned: [4.0089381e-01 2.3377793e-01 1.7155512e-01 7.3547199e-02 5.5353384e-02
    3.5681739e-02 2.2261711e-02 6.9290772e-03 2.4344339e-15 2.4052477e-15]
    Obtaining definitional sets.
    Word(s) found: ['black', 'caucasian', 'asian'], not found: []
    Word(s) found: ['african', 'caucasian', 'asian'], not found: []
    Word(s) found: ['black', 'white', 'asian'], not found: []
    Word(s) found: ['africa', 'america', 'asia'], not found: []
    Word(s) found: ['africa', 'america', 'china'], not found: []
    Word(s) found: ['africa', 'europe', 'asia'], not found: []
    6/6 sets of words were correctly converted to sets of embeddings
    Identifying the bias subspace.
    Obtaining equalize pairs.
    Word(s) found: ['manager', 'executive', 'redneck', 'hillbilly', 'leader', 'farmer'], not found: []
    Word(s) found: ['doctor', 'engineer', 'laborer', 'teacher'], not found: []
    Word(s) found: ['slave', 'musician', 'runner', 'criminal', 'homeless'], not found: []
    3/3 sets of words were correctly converted to sets of embeddings
    Executing Multiclass Hard Debias on glove-twitter-dim=25
    copy argument is True. Transform will attempt to create a copy of the original model. This may fail due to lack of memory.


    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors

    Model copy created successfully.
    Normalizing embeddings.
    Neutralizing embeddings

    100%|██████████| 1193504/1193504 [01:38<00:00, 12108.73it/s]
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    DEBUG:wefe.debias.multiclass_hard_debias:Equalizing embeddings..
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors

    Normalizing embeddings.
    Normalizing embeddings.
    Done!

.. code:: python

    # test with weat
    
    ethnicity_query = Query(
        [
            multiclass_debias_wordsets["white_terms"],
            multiclass_debias_wordsets["black_terms"],
        ],
        [multiclass_debias_wordsets["white_biased_words"], 
        multiclass_debias_wordsets["black_biased_words"]],
        ["european_american_names", "african_american_names"],
        ["white_biased_words", "black_biased_words"],
    )
    
    biased_results = weat.run_query(
        ethnicity_query, model, normalize=True, preprocessors=[{}, {"lowercase": True}],
    )
    debiased_results = weat.run_query(
        ethnicity_query,
        ethnicity_debiased_model,
        normalize=True,
        preprocessors=[{}, {"lowercase": True}],
    )

Absolute value is used here because the closer the value is to zero, the
less biased the model is.

.. code:: python

    import numpy as np
    
    print(
        '| -',
        round(np.abs(debiased_results["weat"]), 3),
        "| < | -",
        round(np.abs(biased_results["weat"]), 3),
        "| =",
        np.abs(debiased_results["weat"]) < np.abs(biased_results["weat"]),
    )


.. parsed-literal::

    | - 0.027 | < | - 0.088 | = True
