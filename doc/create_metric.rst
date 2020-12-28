How to implement your own metric
================================

The following guide will show you how to implement a metric using WEFE.

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

    from ..metrics.base_metric import BaseMetric
     
    class ExampleMetric(BaseMetric):
        metric_template = (2, 1)
        metric_name = 'Example Metric'
        metric_short_name = 'EM'

Implement ``run_query`` method
------------------------------

The second step is to implement ``run_query`` method. This method is in
charge of storing all the operations to calculate the scores from a
``query`` and the ``word_embedding`` model. It must perform 2 basic
operations before executing the mathematical calculations:

Validate the parameters:
~~~~~~~~~~~~~~~~~~~~~~~~

To do this, execute the function :code:``run_query`` from the
``BaseMetric`` class. This call checks the parameters provided to the
``run_query`` and will raise an exception if it finds a problem with
them.

.. code:: python

   # check the types of the provided arguments (only the defaults).
   super().run_query(query, word_embedding, lost_vocabulary_threshold,
                     preprocessor_args, secondary_preprocessor_args,
                     warn_not_found_words, *args, **kwargs)

Transform the Query to Embeddings.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This call transforms all the word sets of a query into embeddings.

.. code:: python

   # transform query word sets into embeddings
   embeddings = word_embedding.get_embeddings_from_query(
       query=query,
       lost_vocabulary_threshold=lost_vocabulary_threshold,
       preprocessor_args=preprocessor_args,
       secondary_preprocessor_args=secondary_preprocessor_args,
       warn_not_found_words=warn_not_found_words)

This step could return either:

-  ``None`` None if for at least one of the word sets in the query there
   are more words without embedding vector than those specified in the
   ``lost_vocabulary_threshold`` parameter (specified as percentage
   float).
-  A tuple otherwise. This tuple contains two values:

   -  A dictionary that maps each target set name to a dictionary
      containing its words and embeddings.
   -  A dictionary that maps each attribute set name to a dictionary
      containing its words and embeddings.

We can illustrate what the outputs of the previous transformation look
like using the following query:

.. code:: python3

     from ..word_embedding_model import WordEmbeddingModel
     from ..query import Query
     from ..utils import load_weat_w2v # a few embeddings of WEAT experiments
     from ..datasets.datasets import load_weat # the word sets of WEAT experiments
      
         
     weat = load_weat()
     model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')
     
     flowers = weat['flowers']
     weapons = weat['weapons']
     pleasant = weat['pleasant_5']
     query = Query([flowers, weapons], [pleasant],
                   ['Flowers', 'Weapons'], ['Pleasant'])
    
    embeddings = model.get_embeddings_from_query(query=query)
    
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

The idea of keeping the words and not just returning the embeddings is
because that there are some metrics that can calculate per-word
measurements and deliver useful information from these.

Using the above, you can already implement the run_query method

.. code:: python3

    from typing import Any, Dict, Union
    
    import numpy as np
    
    from ..metrics.base_metric import BaseMetric
    from ..query import Query
    from ..word_embedding_model import WordEmbeddingModel, PreprocessorArgs
    
    
    class ExampleMetric(BaseMetric):
    
        # replace with the parameters of your metric
        metric_template = (2, 1) # cardinalities of the targets and attributes sets that your metric will accept.
        metric_name = 'Example Metric' 
        metric_short_name = 'EM'
    
        def run_query(self,
                      query: Query,
                      word_embedding: WordEmbeddingModel,
                      # any parameter that you need
                      # ...,
                      lost_vocabulary_threshold: float = 0.2,
                      preprocessor_args: PreprocessorArgs = {
                          'strip_accents': False,
                          'lowercase': False,
                          'preprocessor': None,
                      },
                      secondary_preprocessor_args: PreprocessorArgs = None,
                      warn_not_found_words: bool = False,
                      *args: Any,
                      **kwargs: Any) -> Dict[str, Any]:
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
            
            secondary_preprocessor_args : PreprocessorArgs, optional
                Dictionary with the arguments that specify how the pre-processing of the 
                words will be done, by default {}
                The possible arguments for the function are: 
                - lowercase: bool. Indicates if the words are transformed to lowercase.
                - strip_accents: bool, {'ascii', 'unicode'}: Specifies if the accents of 
                                 the words are eliminated. The stripping type can be 
                                 specified. True uses 'unicode' by default.
                - preprocessor: Callable. It receives a function that operates on each 
                                word. In the case of specifying a function, it overrides 
                                the default preprocessor (i.e., the previous options 
                                stop working).
                , by default { 'strip_accents': False, 'lowercase': False, 'preprocessor': None, }
            
            secondary_preprocessor_args : PreprocessorArgs, optional
                Dictionary with the arguments that specify how the secondary pre-processing 
                of the words will be done, by default None.
                Indicates that in case a word is not found in the model's vocabulary 
                (using the default preprocessor or specified in preprocessor_args), 
                the function performs a second search for that word using the preprocessor 
                specified in this parameter.
    
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
            super().run_query(query, word_embedding, lost_vocabulary_threshold,
                              preprocessor_args, secondary_preprocessor_args,
                              warn_not_found_words, *args, **kwargs)
    
            # transform query word sets into embeddings
            embeddings = word_embedding.get_embeddings_from_query(
                query=query,
                lost_vocabulary_threshold=lost_vocabulary_threshold,
                preprocessor_args=preprocessor_args,
                secondary_preprocessor_args=secondary_preprocessor_args,
                warn_not_found_words=warn_not_found_words)
    
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
            # metric operations. It is recommended to calculate it in another method(s).
            results = calc_metric()        
            
            # You must return query and result. 
            # However, you can return other calculated metrics, metrics by word or metrics by set, etc.
            return {
                    'query_name': query.query_name, # the name of the evaluated query
                    'result': results.metric, # the result of the metric
                    'em': results.metric # result of the calculated metric (recommended)
                    'other_metric' : results.other_metric # Another metric calculated (optional)
                    'another_results' : results.details_by_set # if available, values by word (optional),
                    ...
                }
            """
    

This is what the transformed :code:``target_embeddings_dict`` would look
like:

Implement the logic of the metric
---------------------------------

Suppose we want to implement an extremely simple three-step metric,
where:

1. We calculate the average of all the sets,
2. Then, calculate the cosine distance between the target set averages
   and the attribute average.
3. Subtract these distances.

To do this, we create a new method :code:``_calc_metric`` in which,
using the array of embedding dict objects as input, we will implement
the above.

.. code:: python3

    from ..metrics import BaseMetric
    from ..query import Query
    from ..word_embedding_model import WordEmbeddingModel
    from scipy.spatial import distance
    import numpy as np
    
    
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
                word_embedding: WordEmbeddingModel,
                # any parameter that you need
                # ...,
                lost_vocabulary_threshold: float = 0.2,
                preprocessor_args: PreprocessorArgs = {
                    'strip_accents': False,
                    'lowercase': False,
                    'preprocessor': None,
                },
                secondary_preprocessor_args: PreprocessorArgs = None,
                warn_not_found_words: bool = False,
                *args: Any,
                **kwargs: Any) -> Dict[str, Any]:
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
    
            secondary_preprocessor_args : PreprocessorArgs, optional
                Dictionary with the arguments that specify how the pre-processing of the 
                words will be done, by default {}
                The possible arguments for the function are: 
                - lowercase: bool. Indicates if the words are transformed to lowercase.
                - strip_accents: bool, {'ascii', 'unicode'}: Specifies if the accents of 
                                 the words are eliminated. The stripping type can be 
                                 specified. True uses 'unicode' by default.
                - preprocessor: Callable. It receives a function that operates on each 
                                word. In the case of specifying a function, it overrides 
                                the default preprocessor (i.e., the previous options 
                                stop working).
                , by default { 'strip_accents': False, 'lowercase': False, 'preprocessor': None, }
    
            secondary_preprocessor_args : PreprocessorArgs, optional
                Dictionary with the arguments that specify how the secondary pre-processing 
                of the words will be done, by default None.
                Indicates that in case a word is not found in the model's vocabulary 
                (using the default preprocessor or specified in preprocessor_args), 
                the function performs a second search for that word using the preprocessor 
                specified in this parameter.
    
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
            super().run_query(query, word_embedding, lost_vocabulary_threshold,
                              preprocessor_args, secondary_preprocessor_args,
                              warn_not_found_words, *args, **kwargs)
    
            # transform query word sets into embeddings
            embeddings = word_embedding.get_embeddings_from_query(
                query=query,
                lost_vocabulary_threshold=lost_vocabulary_threshold,
                preprocessor_args=preprocessor_args,
                secondary_preprocessor_args=secondary_preprocessor_args,
                warn_not_found_words=warn_not_found_words)
    
            # if there is any/some set has less words than the allowed limit,
            # return the default value (nan)
            if embeddings is None:
                return {
                    'query_name':
                    query.query_name,  # the name of the evaluated query
                    'result': np.nan,  # the result of the metric
                    'em': np.nan,  # result of the calculated metric (recommended)
                    'other_metric': np.nan,  # another metric calculated (optional)
                    'results_by_word':
                    np.nan,  # if available, values by word (optional)
                    # ...
                }
    
            # get the targets and attribute sets transformed into embeddings.
            target_sets, attribute_sets = embeddings
    
            target_embeddings = list(target_sets.values())
            attribute_embeddings = list(attribute_sets.values())
    
            result = self._calc_metric(target_embeddings, attribute_embeddings)
    
            # return the results.
            return {"query_name": query.query_name, "result": result, 'em': result}

Now, let’s try it out:

.. code:: python3

    from ..query import Query
    from ..utils import load_weat_w2v  # a few embeddings of WEAT experiments
    from ..datasets.datasets import load_weat  # the word sets of WEAT experiments
    
    weat = load_weat()
    model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')
    
    flowers = weat['flowers']
    weapons = weat['weapons']
    pleasant = weat['pleasant_5']
    query = Query([flowers, weapons], [pleasant], ['Flowers', 'Weapons'],
                  ['Pleasant'])
    
    
    results = ExampleMetric().run_query(query, model)
    print(results)

We have completely defined a new metric. Congratulations!

.. warning::

    Some comments regarding the implementation of new metrics:

    - Note that the returned object must necessarily be a ``dict`` instance 
    containing the ``result`` and ``query_name`` key-values. Otherwise
    you will not be able to run query batches using utility functions
    like ``run_queries``.
    - ``run_query`` can receive additional parameters. Simply add them to
    the function signature. These parameters can also be used when
    running the metric from the ``run_queries`` utility function.
    -  We recommend implementing the logic of the metric separated from the
    ``run_query`` function. In other words, implement the logic in a
    ``calc_your_metric`` function that receives the dictionaries with the
    necessary embeddings and parameters.
    -  The file where ``ExampleMetric`` is located can be found inside the
    distances folder of the
    ``repository <https://github.com/dccuchile/wefe/blob/master/wefe/metrics/example_metric.py/>``\ \_.
    

Contribute
----------

If you want to contribute your own metric, please follow the
conventions, document everything, create specific tests for the metric,
and make a pull request to the project’s Github repository. We would
really appreciate it!

You can visit the ``Contributing <contribute.html>``\ \_ section for
more information.
