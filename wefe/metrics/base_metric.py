from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union, NoReturn, Tuple, List
from ..query import Query
from ..word_embedding_model import WordEmbeddingModel


class BaseMetric(ABC):
    """ A base class fot implement any metric.

    It contains several utils for common Metric operations such as checking the
    inputs before executing a query and transforming a queries words into word
    embeddings, among others.

    """
    def __init__(self, metric_template: Tuple[Union[int, str], Union[int,
                                                                     str]],
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
        if not isinstance(metric_template[0], (str, int)) or not isinstance(
                metric_template[1], (str, int)):
            raise TypeError(
                'Both components of template_required must be int or str.'
                ' Given: {}'.format(metric_template))
        if not isinstance(metric_name, str):
            raise TypeError(
                'metric_name must be a str. given: {}'.format(metric_name))
        if not isinstance(metric_short_name, str):
            raise TypeError(
                'metric_short_name must be a str. given: {}'.format(
                    metric_short_name))

        self.metric_template_ = metric_template
        self.metric_name_ = metric_name
        self.metric_short_name_ = metric_short_name

    def _check_input(
        self,
        query: Query,
        word_embedding: WordEmbeddingModel,
        lost_vocabulary_threshold: Union[float, np.float32, np.float64],
        word_preprocessor_options: Dict,
        also_search_for_lowecase: bool,
        warn_filtered_words: bool,
    ) -> None:
        """Checks if the input of a metric is valid.

        Parameters
        ----------
        query : Query
            The query that the method will execute.
        word_embedding : WordEmbeddingModel
            A word embedding model.
        lost_vocabulary_threshold : Union[float, np.float32, np.float64]
            A float number that indicates the acceptable threshold of
            lost words.
        warn_filtered_words : bool
            A bool that indicates if the word to embedding transformation
            will warn about the lost words.

        Raises
        ------
        TypeError
            if query is not instance of Query.
        TypeError
            if word_embedding is not instance of WordEmbeddingModel.
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
            raise TypeError(
                'query parameter must be a Query instance. Given {}'.format(
                    query))

        # check if the word_embedding is a instance of WordEmbeddingModel
        if not isinstance(word_embedding, WordEmbeddingModel):
            raise TypeError(
                'word_embedding must be a WordEmbeddingModel instance. '
                'Given: {}'.format(word_embedding))

        if not isinstance(lost_vocabulary_threshold,
                          (float, np.float, np.float32, np.float64)):
            raise TypeError(
                'lost_vocabulary_threshold must be a float. Given: {}'.format(
                    lost_vocabulary_threshold))

        # templates:

        # check the cardinality of the target sets of the provided query
        if self.metric_template_[0] != 'n' and query.template_[
                0] != self.metric_template_[0]:
            raise Exception(
                'The cardinality of the set of target words of the provided '
                'query ({}) is different from the cardinality required by {}:'
                ' ({})'.format(query.template_[0], self.metric_name_,
                               self.metric_template_[0]))

        # check the cardinality of the attribute sets of the provided query
        if self.metric_template_[1] != 'n' and query.template_[
                1] != self.metric_template_[1]:
            raise Exception(
                'The cardinality of the set of attributes words of the '
                'provided query ({}) is different from the cardinality '
                'required by {}: ({})'.format(query.template_[1],
                                              self.metric_name_,
                                              self.metric_template_[1]))

        if not isinstance(lost_vocabulary_threshold, bool):
            raise TypeError(
                'lost_vocabulary_threshold must be a bool. Given: {}'.format(
                    lost_vocabulary_threshold))

        if not isinstance(word_preprocessor_options, Dict):
            raise TypeError(
                'word_preprocessor_options must be a dictionary. Given: {}'.
                format(word_preprocessor_options))

        if not isinstance(also_search_for_lowecase, bool):
            raise TypeError(
                'also_search_for_lowecase must be a bool. Given: {}'.format(
                    also_search_for_lowecase))

        if not isinstance(warn_filtered_words, bool):
            raise TypeError(
                'warn_filtered_words must be a bool. Given: {}'.format(
                    warn_filtered_words))

    @abstractmethod
    def run_query(
            self,
            query: Query,
            word_embedding: WordEmbeddingModel,
            lost_vocabulary_threshold: float = 0.2,
            word_preprocessor_options: Dict = {
                'remove_word_punctuations': False,
                'translate_words_to_ascii': False,
                'lowercase_words': False,
                'custom_preprocesor': None
            },
            also_search_for_lowecase: bool = False,
            warn_filtered_words: bool = False,
            *args: Any,
            **kwargs: Any) -> Dict:

        self._check_input(query=query,
                          word_embedding=word_embedding,
                          lost_vocabulary_threshold=lost_vocabulary_threshold,
                          word_preprocessor_options=word_preprocessor_options,
                          also_search_for_lowecase=also_search_for_lowecase,
                          warn_filtered_words=warn_filtered_words)

        return {}
