from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable, Dict, List, Union, Tuple
from ..query import Query
from ..word_embedding import WordEmbedding


class BaseMetric(ABC):
    """ A base class fot implement any metric.

    It contains several utils for common Metric operations such as checking the
    inputs before executing a query and transforming a queries words into word
    embeddings, among others.

    """
    def __init__(self, metric_template: Tuple[Union[int, str], Union[int, str]],
                 metric_name: str, metric_short_name: str):
        """Initializes a BaseMetric.

        Parameters
        ----------
        metric_template : Tuple[Union[int, str], Union[int, str]]
            The template (the cardinality of target and attribute sets)
            required for the operation of the metric.
        metric_name : str
            The name of the metric.
        metric_short_name : str
            The initials or short name of the metric.

        Raises
        ------
        TypeError
            If some element of the template is not an integer or string.
        TypeError
            If metric_name is not a string
        TypeError
            If metric_short_name is not a string

        Attributes
        ----------
        metric_template_ : Tuple[Union[int, str], Union[int, str]]
            A tuple that indicates the size of target and attribute sets
            required for the operation of the metric.
        metric_name : str
            The name of the metric.
        metric_short_name:
            A short name of abbreviation of the metric.
        """

        # check types
        if not isinstance(metric_template[0],
                          (str, int)) or not isinstance(metric_template[1], (str, int)):
            raise TypeError('Both components of metric_template should be a int or str,'
                            ' got: {}'.format(metric_template))
        if not isinstance(metric_name, str):
            raise TypeError('metric_name should be a str, got: {}'.format(metric_name))
        if not isinstance(metric_short_name, str):
            raise TypeError(
                'metric_short_name should be a str, got: {}'.format(metric_short_name))

        self.metric_template = metric_template
        self.metric_name = metric_name
        self.metric_short_name = metric_short_name

    def _check_input(
        self,
        query: Query,
        word_embedding: WordEmbedding,
    ) -> None:
        """Checks if the input of a metric is valid.

        Parameters
        ----------
        query : Query
            The query that the method will execute.
        word_embedding : 
            A word embedding model.

        Raises
        ------
        TypeError
            if query is not instance of Query.
        TypeError
            if word_embedding is not instance of .
        TypeError
            if lost_vocabulary_threshold is not a float number.
        TypeError
            if warn_filtered_words is not a bool.
        Exception
            if the metric require different number of target sets than
            the delivered query
        Exception
            if the metric require different number of attribute sets than
            the delivered query
        """

        # check if the query passed is a instance of Query
        if not isinstance(query, Query):
            raise TypeError('query should be a Query instance, got {}'.format(query))

        # check if the word_embedding is a instance of
        if not isinstance(word_embedding, WordEmbedding):
            raise TypeError('word_embedding should be a WordEmbedding instance, '
                            'got: {}'.format(word_embedding))

        # templates:

        # check the cardinality of the target sets of the provided query
        if self.metric_template[0] != 'n' and query.template[0] != self.metric_template[
                0]:
            raise Exception('The cardinality of the set of target words of the \'{}\' '
                            'query does not match with the cardinality required by {}. '
                            'Provided query: {}, metric: {}.'.format(
                                query.query_name, self.metric_name, query.template[0],
                                self.metric_template[0]))

        # check the cardinality of the attribute sets of the provided query
        if self.metric_template[1] != 'n' and query.template[1] != self.metric_template[
                1]:
            raise Exception('The cardinality of the set of attribute words of the '
                            '\'{}\' query does not match with the cardinality '
                            'required by {}. Provided query: {}, metric: {}.'.format(
                                query.query_name, self.metric_name, query.template[1],
                                self.metric_template[1]))

    @abstractmethod
    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbedding,
                  lost_vocabulary_threshold: float = 0.2,
                  preprocessor_options: Dict[str, Union[bool, str, Callable, None]] = {
                      'strip_accents': False,
                      'lowercase': False,
                      'preprocessor': None,
                  },
                  secondary_preprocessor_options: Union[Dict[str, Union[bool, str,
                                                                        Callable, None]],
                                                        None] = None,
                  warn_not_found_words: bool = False,
                  *args: Any,
                  **kwargs: Any) -> Dict[str, Any]:

        self._check_input(query=query, word_embedding=word_embedding)

        return {}