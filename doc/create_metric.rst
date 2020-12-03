================================
How to implement your own metric
================================

The following guide will show you how to implement your own metrics 
using this toolkit.

Create the class 
================

The first step is to create the class that will contain the new metric and 
extend the ``BaseMetric`` class. 
Then, to create the class constructor, you must specify the template 
(explained below), the name of the metric and an abbreviated name or acronym 
for it.

A **template** is a tuple that indicates the number (cardinality) of target 
and attribute word sets that the metric can process. 
It can take integer values as well as the string 'n', which would indicate 
that the metric can accept any amount of that type of set 
(target or attribute words).

.. note::
    Note that this will indicate that all queries that do not comply with the 
    template will be rejected when executed using this metric.

>>> from .base_metric import BaseMetric
>>> 
>>> 
>>> class ExampleMetric(BaseMetric):
>>>     def __init__(self):
>>> 
>>>         # define the required parameters.
>>>         template_needed = (2, 1)
>>>         metric_name = 'Example Metric'
>>>         metric_short_name = 'EM'
>>>         # initialize the super class.
>>>         super().__init__(template_needed, metric_name, metric_short_name)


An examples of template 

>>> # two target sets and one attribute set required to execute this metric.
>>> template_1 = (2, 1)
>>>
>>> # two target sets and two attribute set required to execute this metric.
>>> template_2 = (2, 2)
>>>
>>> # two target sets and one attribute set required to execute this metric.
>>> template_1 = ('n', 1)

Implement :code:`run_query` method
==================================

Next, you must implement the :code:`run_query` method. 
This method will evaluate the metric on a given embedding model.
It must perform 2 basic operations before executing anything. 

1. Validate the inputs and Transforming :code:`query` to embeddings: 

    To do this, execute the function :code:`self._get_embeddings_from_query` 
    using the provided parameters.  

    First, it validates that the parameters 
    :code:`query`, :code:`word_embedding`, 
    :code:`lost_vocabulary_threshold` and 
    :code:`warn_filtered_words` are of type :code:`Query`, 
    :code:``, ``bool`` and ``bool`` respectively.
    It will raise an exception if it finds a problem with the parameters.

    Second, if no error triggered an exception, it converts all word sets 
    into Word Embeddings.
    This process could return either: 
    
        - ``None`` None if for at least one of the word sets in the query there
          are more words without embedding vector than those specified in the 
          ``lost_vocabulary_threshold`` parameter (specified as percentage 
          float).  
        - A tuple is otherwise returned. This tuple will contain in the first 
          element an array of dictionaries with the embeddings of each target 
          set, and in the second element, an array of dictionaries with the 
          embeddings of each attribute set. Each dictionary will be a mapping 
          between words and their corresponding embedding vectors.        

2. Check if the array of embeddings is None when there is a word set in which 
more than the threshold percentage of words does not have an embedding.

>>> from .base_metric import BaseMetric
>>> from ..query import Query
>>> from ..word_embedding import 
>>>
>>>
>>> class ExampleMetric(BaseMetric):
>>>     def __init__(self):
>>> 
>>>         template_needed = (2, 1)
>>>         metric_name = 'Example Metric'
>>>         metric_short_name = 'EM'
>>>         super().__init__(template_needed, metric_name, metric_short_name)
>>> 
>>>     def run_query(self, query: Query, word_embedding: ,
>>>                   lost_vocabulary_threshold: float = 0.2,
>>>                   warn_filtered_words: bool = True):
>>> 
>>> 
>>>         # check the inputs and get the embeddings.
>>>         embeddings = self._get_embeddings_from_query(
>>>             query, word_embedding, warn_filtered_words,
>>>             lost_vocabulary_threshold)
>>>
>>>         # if there's any set that has a percentage fewer words than the threshold,
>>>         # return the default value (nan)
>>>         if embeddings is None:
>>>             return {'query_name': query.query_name, 'result': np.nan}


We can illustrate what the outputs of the previous transformation look like 
using the following query:

>>> from wefe.word_embedding import 
>>> from wefe.query import Query
>>> from wefe.utils import load_weat_w2v # a few embeddings of WEAT experiments
>>> from wefe.datasets.datasets import load_weat # the word sets of WEAT experiments
>>>  
>>>     
>>> weat = load_weat()
>>> model = (load_weat_w2v(), 'weat_w2v', '')
>>> 
>>> flowers = weat['flowers'][0:4]
>>> weapons = weat['weapons'][0:4]
>>> pleasant = weat['pleasant_5'][0:4]
>>> query = Query([flowers, weapons], [pleasant],
>>>               ['Flowers', 'Weapons'], ['Pleasant'])
>>>
>>> # Execute the transformation
>>> target_embeddings_dict, attribute_embeddings_dict = ExampleMetric()._get_embeddings_from_query(
    query, model)


This is what the transformed :code:`target_embeddings_dict` would look like:

>>> [{'aster': array([-0.22167969,  0.52734375,  0.01745605, ...], dtype=float32),
>>>   'clover': array([-0.03442383,  0.19042969, -0.17089844, ...], dtype=float32),
>>>   'hyacinth': array([-0.01391602,  0.3828125 , -0.21679688, ...], dtype=float32),
>>>   'marigold': array([-0.27539062,  0.1484375 ,  0.04516602, ...], dtype=float32),
>>>   'poppy': array([ 0.19433594, -0.14257812, -0.07324219, ...], dtype=float32)},
>>>  {'arrow': array([ 0.18164062,  0.125     , -0.12792969, ...], dtype=float32),
>>>   'club': array([-0.04907227, -0.07421875, -0.0390625, ... ], dtype=float32),
>>>   'gun': array([0.05566406, 0.15039062, 0.33398438, ...], dtype=float32),
>>>   'missile': array([ 4.7874451e-04,  5.1953125e-01, -1.3809204e-03, ...], dtype=float32),
>>>   'spear': array([ 0.1875    , -0.0008316 , -0.11816406, ...], dtype=float32)}]

This is what the transformed :code:`attribute_embeddings_dict` would look like:

>>> [{'caress': array([ 0.2578125 , -0.22167969,  0.11669922, ...], dtype=float32),
>>>   'freedom': array([ 0.26757812, -0.078125  ,  0.09326172, ...], dtype=float32),
>>>   'health': array([-0.07421875,  0.11279297,  0.09472656, ...], dtype=float32),
>>>   'love': array([ 0.10302734, -0.15234375,  0.02587891, ...], dtype=float32),
>>>   'peace': array([0.15722656, 0.26171875, 0.27734375, ...], dtype=float32)}]


.. note::
    The idea of keeping the words and not just returning the 
    embeddings is because that there are some metrics that can 
    calculate per-word measurements and deliver useful information from these.


Then, based on the above, you can implement your new metric.


Implement the logic of the metric
=================================


Suppose we want to implement an extremely simple three-step metric, where:

1. We calculate the average of all the sets,
2. Then, calculate the cosine distance between the target set averages and the 
   attribute average.
3. Subtract these distances.

To do this, we create a new method :code:`__calc_metric` in which, using
the array of embedding dict objects as input, we will implement the above.

>>> from .base_metric import BaseMetric
>>> from ..query import Query
>>> from ..word_embedding import 
>>> from scipy.spatial import distance
>>> import numpy as np
>>> 
>>> 
>>> class ExampleMetric(BaseMetric):
>>>     def __init__(self):
>>> 
>>>         template_needed = (2, 1)
>>>         metric_name = 'Example Metric'
>>>         metric_short_name = 'EM'
>>>         super().__init__(template_needed, metric_name, metric_short_name)
>>> 
>>>     def __calc_metric(self, target_embeddings, attribute_embeddings):
>>>         """Calculates the metric.
>>>         
>>>         Parameters
>>>         ----------
>>>         target_embeddings : np.array
>>>             An array with dicts. Each dict represents an target set. A dict is composed with a word and its embedding as key, value respectively.
>>>         attribute_embeddings : np.array
>>>             An array with dicts. Each dict represents an attribute set. A dict is composed with a word and its embedding as key, value respectively.
>>>         
>>>         Returns
>>>         -------
>>>         np.float
>>>             The value of the calculated metric.
>>>         """
>>> 
>>>         # get the embeddings from the dicts
>>>         target_embeddings_0 = np.array(list(target_embeddings[0].values()))
>>>         target_embeddings_1 = np.array(list(target_embeddings[1].values()))
>>> 
>>>         attribute_embeddings_0 = np.array(
>>>             list(attribute_embeddings[0].values()))
>>> 
>>>         # calculate the average embedding by target and attribute set.
>>>         target_embeddings_0_avg = np.mean(target_embeddings_0, axis=0)
>>>         target_embeddings_1_avg = np.mean(target_embeddings_1, axis=0)
>>>         attribute_embeddings_0_avg = np.mean(attribute_embeddings_0, axis=0)
>>> 
>>>         # calculate the distances between the target sets and the attribute set
>>>         dist_target_0_attr = distance.cosine(target_embeddings_0_avg,
>>>                                              attribute_embeddings_0_avg)
>>>         dist_target_1_attr = distance.cosine(target_embeddings_1_avg,
>>>                                              attribute_embeddings_0_avg)
>>> 
>>>         # subtract the distances
>>>         metric_result = dist_target_0_attr - dist_target_1_attr
>>>         return metric_result
>>> 
>>>     def run_query(self, query: Query, word_embedding: ,
>>>                   lost_vocabulary_threshold: float = 0.2,
>>>                   warn_filtered_words: bool = True):
>>> 
>>>         # check the inputs and get the embeddings
>>>         embeddings = self._get_embeddings_from_query(
>>>             query, word_embedding, warn_filtered_words,
>>>             lost_vocabulary_threshold)
>>> 
>>>         # if there is any/some set has less words than the allowed limit, return the default value (nan)
>>>         if embeddings is None:
>>>             return {'query_name': query.query_name, 'result': np.nan}
>>> 
>>>         # separate the embedding tuple
>>>         target_embeddings, attribute_embeddings = embeddings
>>> 
>>>         # execute the metric
>>>         metric_result = self.__calc_metric(target_embeddings,
>>>                                            attribute_embeddings)
>>> 
>>>         # return the results.
>>>         return {
>>>             "query_name": query.query_name,
>>>             "result": metric_result,
>>>         }

We have completely defined a new metric.
Congratulations!

.. note::
    Some comments regarding the implementation of new metrics:

    - Note that the returned object must necessarily be a ``dict`` instance 
      containing the ``result`` and ``query_name`` key-values. Otherwise you 
      will not be able to run query batches using utility functions like 
      ``run_queries``.
    - ``run_query`` can receive additional parameters. Simply add them to the 
      function signature. These parameters can also be used when running the 
      metric from the ``run_queries`` utility function.
    - We recommend implementing the logic of the metric separated from the 
      ``run_query`` function. In other words, implement the logic in a 
      ``calc_your_metric`` function that receives the dictionaries with the 
      necessary embeddings and parameters.
    - The file where ``ExampleMetric`` is located can be found inside the 
      distances folder of the `repository <https://github.com/dccuchile/wefe/blob/master/wefe/metrics/example_metric.py/>`_. 

Contribute
==========

If you want to contribute your own metric, please follow the conventions, 
document everything, create specific tests for the metric, and make a pull 
request to the project's Github repository. 
We would really appreciate it! 

You can visit the `Contributing <contribute.html>`_ section for more information.