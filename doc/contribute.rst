============
Contributing
============

There are many ways to contribute to the library: 

- Implementing new metrics. 
- Implementing new mitigation methods.
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

-----


How to implement your own metric
================================

The following guide is intended to show how to implement a metric using WEFE.
You can find a notebook version of this tutorial at the following 
`link <https://github.com/dccuchile/wefe/blob/master/wefe/examples/Contributing.ipynb/>`__.

Create the class
----------------

The first step is to create the class that will contain the metric. This
class must extend the ``BaseMetric`` class.

In the new class you must specify the template (explained below), the
name and an abbreviated name or acronym for the metric as class
variables.

A **template** is a tuple that defines the cardinality of the tagret and
attribute sets of a query that can be accepted by the metric. It can
take integer values, which require that the target or attribute sets
have that cardinality or ‘n’ in case the metric can operate with 1 or
more word sets. Note that this will indicate that all queries that do
not comply with the template will be rejected when executed using this
metric.

Below are some examples of templates:

.. code:: python3

    # two target sets and one attribute set required to execute this metric.
    template_1 = (2, 1)
    
    # two target sets and two attribute set required to execute this metric.
    template_2 = (2, 2)
    
    # one or more (unlimited) target sets and one attribute set required to execute this metric.
    template_3 = ('n', 1)

Once the template is defined, you can create the metric according to the
following code scheme:

.. code:: python3

    from wefe.metrics.base_metric import BaseMetric
        
    class ExampleMetric(BaseMetric):
        metric_template = (2, 1)
        metric_name = 'Example Metric'
        metric_short_name = 'EM'

Implement ``run_query`` method
------------------------------

The second step is to implement ``run_query`` method. This method is in
charge of coordinate all the operations to calculate the scores from a
``query`` and the ``word_embedding`` model. It must perform 2 basic
operations before executing the mathematical calculations:

Validate the parameters
~~~~~~~~~~~~~~~~~~~~~~~

This call checks the main parameters provided to the ``run_query`` and will raise an 
exception if it finds a problem with them.

.. code:: python

    # check the types of the provided arguments.
    self._check_input(query, model)

Transform the Query to Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This call transforms all the word sets of a query into embeddings.

.. code:: python

    # transform query word sets into embeddings
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        lost_vocabulary_threshold=lost_vocabulary_threshold,
        preprocessors=preprocessors,
        strategy=strategy,
        normalize=normalize,
        warn_not_found_words=warn_not_found_words,
    )

This step could return either:

-   ``None`` if any of the sets lost percentage more words than the number of words 
    allowed by ``lost_vocabulary_threshold`` parameter (specified as percentage
    float). In this case the metric would be expected to return nan in its results.

.. code:: python

    # if there is any/some set has less words than the allowed limit,
    # return the default value (nan)
    if embeddings is None:
        return {
            "query_name": query.query_name,
            "result": np.nan,
            "metrica_default_value": np.nan,
        }


-  A tuple otherwise. This tuple contains two values:

    -  A dictionary that maps each target set name to a dictionary containing its words and embeddings.
    -  A dictionary that maps each attribute set name to a dictionary containing its words and embeddings.

We can illustrate what the outputs of the previous transformation look
like using the following query:

.. code:: python3

    from wefe.word_embedding_model import WordEmbeddingModel
    from wefe.query import Query
    from wefe.utils import load_test_model # a few embeddings of WEAT experiments
    from wefe.datasets.datasets import load_weat # the word sets of WEAT experiments
    from wefe.preprocessing import get_embeddings_from_query
    
        
    weat = load_weat()
    model = load_test_model()
    
    flowers = weat['flowers']
    weapons = weat['weapons']
    pleasant = weat['pleasant_5']
    query = Query([flowers, weapons], [pleasant],
                ['Flowers', 'Weapons'], ['Pleasant'])
    
    embeddings = get_embeddings_from_query(
        model=model,
        query=query,
        # other params...
    )
    target_sets, attribute_sets = embeddings
    

If you inspect ``target_sets``, it would look like the following
dictionary:

.. code:: python

    {
        'Flowers': {
            'aster': array([-0.22167969, 0.52734375, 0.01745605, ...], dtype=float32),
            'clover': array([-0.03442383, 0.19042969, -0.17089844, ...], dtype=float32),
            'hyacinth': array([-0.01391602, 0.3828125, -0.21679688, ...], dtype=float32),
            ...
        },
        'Weapons': {
            'arrow': array([0.18164062, 0.125, -0.12792969. ...], dtype=float32),
            'club': array([-0.04907227, -0.07421875, -0.0390625, ...], dtype=float32),
            'gun': array([0.05566406, 0.15039062, 0.33398438, ...], dtype=float32),
            'missile': array([4.7874451e-04, 5.1953125e-01, -1.3809204e-03, ...], dtype=float32),
            ...
        }
    }

And ``attribute_sets`` would look like:

.. code:: python

    {
        'Pleasant': {
            'caress': array([0.2578125, -0.22167969, 0.11669922], dtype=float32),
            'freedom': array([0.26757812, -0.078125, 0.09326172], dtype=float32),
            'health': array([-0.07421875, 0.11279297, 0.09472656], dtype=float32),
            ...
        }
    }

The idea of keeping a mapping between set names, words and their embeddings is that
there are some metrics that can calculate sub-metrics at different levels and that can
be useful for further use.

Example Metric
~~~~~~~~~~~~~~

Using the steps previously seen, a sample metric is implemented:

.. code:: python3

    from typing import Any, Dict, Union, List, Callable

    import numpy as np
    
    from wefe.metrics.base_metric import BaseMetric
    from wefe.query import Query
    from wefe.word_embedding_model import WordEmbeddingModel, PreprocessorArgs
    
    
    class ExampleMetric(BaseMetric):
    
        # replace with the parameters of your metric
        metric_template = (2, 1) # cardinalities of the targets and attributes sets that your metric will accept.
        metric_name = 'Example Metric' 
        metric_short_name = 'EM'
    
        def run_query(
            self,
            query: Query,
            model: WordEmbeddingModel,
            lost_vocabulary_threshold: float = 0.2,
            preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
            strategy: str = "first",
            normalize: bool = False,
            warn_not_found_words: bool = False,
            *args: Any,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            """Calculate the Example Metric metric over the provided parameters.
    
            Parameters
            ----------
            query : Query
                A Query object that contains the target and attribute word sets to 
                be tested.
    
            word_embedding : WordEmbeddingModel
                A WordEmbeddingModel object that contains certain word embedding 
                pretrained model.
            
            lost_vocabulary_threshold : float, optional
                Specifies the proportional limit of words that any set of the query is
                allowed to lose when transforming its words into embeddings.
                In the case that any set of the query loses proportionally more words
                than this limit, the result values will be np.nan, by default 0.2
    
            preprocessors : List[Dict[str, Union[str, bool, Callable]]]
                A list with preprocessor options.
    
                A ``preprocessor`` is a dictionary that specifies what processing(s) are
                performed on each word before its looked up in the model vocabulary.
                For example, the ``preprocessor``
                ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
                and remove the accent from each word before searching for them in the
                model vocabulary. Note that an empty dictionary ``{}`` indicates that no
                preprocessing is done.
    
                The possible options for a preprocessor are:
    
                *   ``lowercase``: ``bool``. Indicates that the words are transformed to
                    lowercase.
                *   ``uppercase``: ``bool``. Indicates that the words are transformed to
                    uppercase.
                *   ``titlecase``: ``bool``. Indicates that the words are transformed to
                    titlecase.
                *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
                    the accents of the words are eliminated. The stripping type can be
                    specified. True uses ‘unicode’ by default.
                *   ``preprocessor``: ``Callable``. It receives a function that operates
                    on each word. In the case of specifying a function, it overrides the
                    default preprocessor (i.e., the previous options stop working).
    
                A list of preprocessor options allows to search for several
                variants of the words into the model. For example, the preprocessors
                ``[{}, {"lowercase": True, "strip_accents": True}]``
                ``{}`` allows first to search for the original words in the vocabulary of the model. 
                In case some of them are not found, ``{"lowercase": True, "strip_accents": True}`` 
                is executed on these words and then they are searched in the model vocabulary.
    
            strategy : str, optional
                The strategy indicates how it will use the preprocessed words: 'first' will
                include only the first transformed word found. all' will include all
                transformed words found, by default "first".
    
            normalize : bool, optional
                True indicates that embeddings will be normalized, by default False
    
            warn_not_found_words : bool, optional
                Specifies if the function will warn (in the logger)
                the words that were not found in the model's vocabulary
                , by default False.
    
            Returns
            -------
            Dict[str, Any]
                A dictionary with the query name, the resulting score of the metric, 
                and other scores.
            """
            # check the types of the provided arguments (only the defaults).
            self._check_input(query, model)
    
            # transform query word sets into embeddings
            embeddings = get_embeddings_from_query(
                model=model,
                query=query,
                lost_vocabulary_threshold=lost_vocabulary_threshold,
                preprocessors=preprocessors,
                strategy=strategy,
                normalize=normalize,
                warn_not_found_words=warn_not_found_words,
            )
    
            # if there is any/some set has less words than the allowed limit,
            # return the default value (nan)
            if embeddings is None:
                return {
                    'query_name': query.query_name, # the name of the evaluated query
                    'result': np.nan, # the result of the metric
                    'em': np.nan, # result of the calculated metric (recommended)
                    'other_metric' : np.nan, # another metric calculated (optional)
                    'results_by_word' : np.nan, # if available, values by word (optional)
                    # ...
                }
    
            # get the targets and attribute sets transformed into embeddings.
            target_sets, attribute_sets = embeddings
    
            # commonly, you only will need the embeddings of the sets.
            # this can be obtained by using:
            target_embeddings = list(target_sets.values())
            attribute_embeddings = list(attribute_sets.values())
    
            
            """
            # From here, the code can vary quite a bit depending on what you need.
            # It is recommended to calculate the metric operations in another method(s).
            results = calc_metric()        
            
            # The final step is to return query and result. 
            # You can return other scores, metrics by word or metrics by set, etc.
            return {
                    'query_name': query.query_name, # the name of the evaluated query
                    'result': results.metric, # the result of the metric
                    'em': results.metric # result of the calculated metric (recommended)
                    'other_metric' : results.other_metric # Another metric calculated (optional)
                    'another_results' : results.details_by_set # if available, values by word (optional),
                    ...
                }
            """
    


Implement the logic of the metric
---------------------------------

Suppose we want to implement an extremely simple three-step metric,
where:

1.  We calculate the average of all the sets,
2.  Then, calculate the cosine distance between the target set averages
    and the attribute average.
3.  Subtract these distances.

To do this, we create a new method :code:``_calc_metric`` in which,
using the array of embedding dict objects as input, we will implement
the above.

.. code:: python3

    from typing import Any, Dict, Union, List, Callable

    from scipy.spatial import distance
    import numpy as np
    
    from wefe.metrics import BaseMetric
    from wefe.query import Query
    from wefe.word_embedding_model import WordEmbeddingModel
    from wefe.preprocessing import get_embeddings_from_query
    
    class ExampleMetric(BaseMetric):
    
        # replace with the parameters of your metric
        metric_template = (
            2, 1
        )  # cardinalities of the targets and attributes sets that your metric will accept.
        metric_name = 'Example Metric'
        metric_short_name = 'EM'
    
        def _calc_metric(self, target_embeddings, attribute_embeddings):
            """Calculates the metric.
    
             Parameters
             ----------
             target_embeddings : np.array
                 An array with dicts. Each dict represents an target set. 
                 A dict is composed with a word and its embedding as key, value respectively.
             attribute_embeddings : np.array
                 An array with dicts. Each dict represents an attribute set. 
                 A dict is composed with a word and its embedding as key, value respectively.
    
             Returns
             -------
             np.float
                 The value of the calculated metric.
             """
    
            # get the embeddings from the dicts
            target_embeddings_0 = np.array(list(target_embeddings[0].values()))
            target_embeddings_1 = np.array(list(target_embeddings[1].values()))
    
            attribute_embeddings_0 = np.array(
                list(attribute_embeddings[0].values()))
    
            # calculate the average embedding by target and attribute set.
            target_embeddings_0_avg = np.mean(target_embeddings_0, axis=0)
            target_embeddings_1_avg = np.mean(target_embeddings_1, axis=0)
            attribute_embeddings_0_avg = np.mean(attribute_embeddings_0, axis=0)
    
            # calculate the distances between the target sets and the attribute set
            dist_target_0_attr = distance.cosine(target_embeddings_0_avg,
                                                 attribute_embeddings_0_avg)
            dist_target_1_attr = distance.cosine(target_embeddings_1_avg,
                                                 attribute_embeddings_0_avg)
    
            # subtract the distances
            metric_result = dist_target_0_attr - dist_target_1_attr
            return metric_result
    
        def run_query(
            self,
            query: Query,
            model: WordEmbeddingModel,
            lost_vocabulary_threshold: float = 0.2,
            preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
            strategy: str = "first",
            normalize: bool = False,
            warn_not_found_words: bool = False,
            *args: Any,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            """Calculate the Example Metric metric over the provided parameters.
    
            Parameters
            ----------
            query : Query
                A Query object that contains the target and attribute word sets to 
                be tested.
    
            word_embedding : WordEmbeddingModel
                A WordEmbeddingModel object that contains certain word embedding 
                pretrained model.
            
            lost_vocabulary_threshold : float, optional
                Specifies the proportional limit of words that any set of the query is
                allowed to lose when transforming its words into embeddings.
                In the case that any set of the query loses proportionally more words
                than this limit, the result values will be np.nan, by default 0.2
    
            preprocessors : List[Dict[str, Union[str, bool, Callable]]]
                A list with preprocessor options.
    
                A ``preprocessor`` is a dictionary that specifies what processing(s) are
                performed on each word before its looked up in the model vocabulary.
                For example, the ``preprocessor``
                ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
                and remove the accent from each word before searching for them in the
                model vocabulary. Note that an empty dictionary ``{}`` indicates that no
                preprocessing is done.
    
                The possible options for a preprocessor are:
    
                *   ``lowercase``: ``bool``. Indicates that the words are transformed to
                    lowercase.
                *   ``uppercase``: ``bool``. Indicates that the words are transformed to
                    uppercase.
                *   ``titlecase``: ``bool``. Indicates that the words are transformed to
                    titlecase.
                *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
                    the accents of the words are eliminated. The stripping type can be
                    specified. True uses ‘unicode’ by default.
                *   ``preprocessor``: ``Callable``. It receives a function that operates
                    on each word. In the case of specifying a function, it overrides the
                    default preprocessor (i.e., the previous options stop working).
    
                A list of preprocessor options allows to search for several
                variants of the words into the model. For example, the preprocessors
                ``[{}, {"lowercase": True, "strip_accents": True}]``
                ``{}`` allows first to search for the original words in the vocabulary of the model. 
                In case some of them are not found, ``{"lowercase": True, "strip_accents": True}`` 
                is executed on these words and then they are searched in the model vocabulary.
    
            strategy : str, optional
                The strategy indicates how it will use the preprocessed words: 'first' will
                include only the first transformed word found. all' will include all
                transformed words found, by default "first".
    
            normalize : bool, optional
                True indicates that embeddings will be normalized, by default False
    
            warn_not_found_words : bool, optional
                Specifies if the function will warn (in the logger)
                the words that were not found in the model's vocabulary
                , by default False.
    
            Returns
            -------
            Dict[str, Any]
                A dictionary with the query name, the resulting score of the metric, 
                and other scores.
            """
            # check the types of the provided arguments (only the defaults).
            self._check_input(query, model)
    
            # transform query word sets into embeddings
            embeddings = get_embeddings_from_query(
                model=model,
                query=query,
                lost_vocabulary_threshold=lost_vocabulary_threshold,
                preprocessors=preprocessors,
                strategy=strategy,
                normalize=normalize,
                warn_not_found_words=warn_not_found_words,
            )
    
            # if there is any/some set has less words than the allowed limit,
            # return the default value (nan)
            if embeddings is None:
                return {
                    'query_name': query.query_name, # the name of the evaluated query
                    'result': np.nan, # the result of the metric
                    'em': np.nan, # result of the calculated metric (recommended)
                    'other_metric' : np.nan, # another metric calculated (optional)
                    'results_by_word' : np.nan, # if available, values by word (optional)
                    # ...
                }
    
            # get the targets and attribute sets transformed into embeddings.
            target_sets, attribute_sets = embeddings
    
            # commonly, you only will need the embeddings of the sets.
            # this can be obtained by using:
            target_embeddings = list(target_sets.values())
            attribute_embeddings = list(attribute_sets.values())
    
            result = self._calc_metric(target_embeddings, attribute_embeddings)
    
            # return the results.
            return {"query_name": query.query_name, "result": result, 'em': result}

Now, let’s try it out:

.. code:: python3

    from wefe.query import Query
    from wefe.utils import load_weat_w2v  # a few embeddings of WEAT experiments
    from wefe.datasets.datasets import load_weat  # the word sets of WEAT experiments
    
    weat = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')
    
    flowers = weat['flowers']
    weapons = weat['weapons']
    pleasant = weat['pleasant_5']
    query = Query([flowers, weapons], [pleasant], ['Flowers', 'Weapons'],
                    ['Pleasant'])
    
    
    results = ExampleMetric().run_query(query, model)
    print(results)

.. parsed-literal::

    {'query_name': 'Flowers and Weapons wrt Pleasant', 'result': -0.10210171341896057, 'em': -0.10210171341896057}
    

We have completely defined a new metric. Congratulations!

**Note**

Some comments regarding the implementation of new metrics:

-   Note that the returned object must necessarily be a ``dict`` instance
    containing the ``result`` and ``query_name`` key-values. Otherwise
    you will not be able to run query batches using utility functions
    like ``run_queries``.
-   ``run_query`` can receive additional parameters. Simply add them to
    the function signature. These parameters can also be used when
    running the metric from the ``run_queries`` utility function.
-   We recommend implementing the logic of the metric separated from the
    ``run_query`` function. In other words, implement the logic in a
    ``calc_your_metric`` function that receives the dictionaries with the
    necessary embeddings and parameters.
-   The file where ``ExampleMetric`` is located can be found inside the
    distances folder of the
    `repository <https://github.com/dccuchile/wefe/blob/master/wefe/metrics/example_metric.py/>`__.

    
Mitigation Method Implementation Guide
======================================


The main idea when implementing a mitigation method is that it has to follow the logic
of the transformations in scikit-learn. 
That is, you must separate the logic of the calculation of the mitigation 
transformation (`fit`) with the application of the transformation on the model 
(`transform`).

In practical terms, every WEFE transformation must extend the `BaseDebias` class. 
`BaseDebias` has two abstract methods that must be implemented: `fit` and `transform`.


Fit
---


`fit` is the method in charge of calculating the bias mitigation transformation 
that will be subsequently applied to the model.
`BaseDebias` implements it as an abstract method that requires only one argument: 
`model`, which expects a `WordEmbeddingModel` instance.

.. code:: python3

    @abstractmethod
    def fit(
        self,
        model: WordEmbeddingModel,
        **fit_params,
    ) -> "BaseDebias":
        """Fit the transformation.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        """
        raise NotImplementedError()


The idea of requesting model at this point is that the calculation of the 
transformation commonly requires some words from the model vocabulary.

As each bias mitigation method is different, it is expected that these can receive more 
parameters than those listed above. In, `HardDebias`, `fit` is defined using the default
parameter `model` plus `definitional_pairs` and `equalize_pairs`, which are 
specific to `HardDebias`:

.. code:: python3

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        equalize_pairs: Optional[Sequence[Sequence[str]]] = None,
        **fit_params,
    ) -> BaseDebias:
        """Compute the bias direction and obtains the equalize embedding pairs.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        definitional_pairs : Sequence[Sequence[str]]
            A sequence of string pairs that will be used to define the bias direction.
            For example, for the case of gender debias, this list could be [['woman',
            'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ...].
        equalize_pairs : Optional[Sequence[Sequence[str]]], optional
            A list with pairs of strings which will be equalized.
            In the case of passing None, the equalization will be done over the word
            pairs passed in definitional_pairs,
            by default None.
        criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g. 'Gender'. This will indicate the name of the model returning transform,
            by default None

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """
        self._check_sets_size(definitional_pairs, "definitional")
        self.definitional_pairs_ = definitional_pairs

        # ------------------------------------------------------------------------------
        # Obtain the embedding of each definitional pairs.
        if self.verbose:
            print("Obtaining definitional pairs.")
        self.definitional_pairs_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=self.verbose,
            normalize=True,
            verbose=self.verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        if self.verbose:
            print("Identifying the bias subspace.")

        self.pca_ = self._identify_bias_subspace(
            self.definitional_pairs_embeddings_, self.verbose,
        )
        self.bias_direction_ = self.pca_.components_[0]
        # code was cut for simplicity.
        # you can visit the missing code in the file debias/HardDebias
        ...
        return self

.. note::

Note that `get_embeddings_from_sets` is used to transform word sets to embeddings sets. 
This function, as well as the one to transform queries to embeddings, are available 
in the `preprocessing` module.

Once fit has calculated the transformation, the method should return self.



Transform
---------

This method is intended to implement the application of the transformation calculated
in `fit` on the embedding model.It must always receive the same 4 arguments:

- `model`: The model on which the transformation will be applied
- `target`: A set of words or None. If it is specified, the debias method will be performed
  only on the word embeddings of this set. In the case of provide `None`, the
  debias will be performed on all words (except those specified in ignore).
  by default `None`.
- `ignore`: A set of words or None. If target is `None` and a set of words is specified 
- in ignore, the debias method will perform the debias in all words except those 
- specified in this set, by default `None`.
- `copy`: If `True`, the debias will be performed on a copy of the model.
  If `False`, the debias will be applied on the same model delivered, causing
  its vectors to mutate.

.. code:: python

    @abstractmethod
    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Perform the debiasing method over the model provided.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be performed
            only on the word embeddings of this set. In the case of provide `None`, the
            debias will be performed on all words (except those specified in ignore).
            by default `None`.
        ignore : Optional[List[str]], optional
            If target is `None` and a set of words is specified in ignore, the debias
            method will perform the debias in all words except those specified in this
            set, by default `None`.
        copy : bool, optional
            If `True`, the debias will be performed on a copy of the model.
            If `False`, the debias will be applied on the same model delivered, causing
            its vectors to mutate.
            **WARNING:** Setting copy with `True` requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may rise
            `MemoryError`, by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased word embedding model.
        """
        raise NotImplementedError()

As can be seen, the embeddings that will be modified by the transformation will 
be determined by the words delivered in the `target` and `ignore` sets or the 
absence of both (apply on all words).
The idea is that this convention is maintained during the creation of a new debias 
method.

Some useful initial checks and operations for this method:

- The arguments can be checked through the `_check_transform_args` `BaseDebias` method.
- You can also check whether the method is trained or not using the `check_is_fitted` 
  method. This is a wrapper of the original scikit-learn that can be imported from the 
  utils module.
- In case `copy` argument is `True`, you must duplicate the model and work on the 
  replica. It is recommended to use `deepcopy` of the `copy` module for such purposes.

The following code segment (obtained from `HardDebias`) shows an example of how to 
execute the points mentioned above:

.. code:: python

    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
        ) -> WordEmbeddingModel:
        """Execute hard debias over the provided model.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be performed
            only on the word embeddings of this set. In the case of provide `None`, the
            debias will be performed on all words (except those specified in ignore).
            by default `None`.
        ignore : Optional[List[str]], optional
            If target is `None` and a set of words is specified in ignore, the debias
            method will perform the debias in all words except those specified in this
            set, by default `None`.
        copy : bool, optional
            If `True`, the debias will be performed on a copy of the model.
            If `False`, the debias will be applied on the same model delivered, causing
            its vectors to mutate.
            **WARNING:** Setting copy with `True` requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may rise
            `MemoryError`, by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.
        """
        # ------------------------------------------------------------------------------
        # Check types and if the method is fitted

        self._check_transform_args(
            model=model, target=target, ignore=ignore, copy=copy,
        )

        # check if the following attributes exists in the object.
        check_is_fitted(
            self,
            [
                "definitional_pairs_",
                "definitional_pairs_embeddings_",
                "pca_",
                "bias_direction_",
            ],
        )

        # Copy
        if copy:
            print(
                "Copy argument is True. Transform will attempt to create a copy "
                "of the original model. This may fail due to lack of memory."
            )
            model = deepcopy(model)
            print("Model copy created successfully.")

        else:
            print(
                "copy argument is False. The execution of this method will mutate "
                "the original model."
            )

Unfortunately it is impossible to cover much more without losing generality.
However, we recommend to check the code structure shown in HardDebias or 
MulticlassHardDebias to guide you through the process of implementing a new 
mitigation and use these classes as a reference to implement a new debias method. 
You can also open an issue in the repository to comment on any questions you may have
in the implementation.