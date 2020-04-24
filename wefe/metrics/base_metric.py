from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
import logging
import numpy as np
from typing import Union, NoReturn, Tuple, List


class BaseMetric(object):
    """ A base class fot implement any metric.

    It contains several utils for common Metric operations such as check the
    inputs before execute a query and transform a queries words into word
    embeddings, among others.

    """
    def __init__(self,
                 metric_template: Tuple[Union[int, str], Union[int, str]],
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
            self, query: Query, word_embedding: WordEmbeddingModel,
            lost_vocabulary_threshold: Union[float, np.float32, np.float64],
            warn_filtered_words: bool) -> NoReturn:
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

        if not isinstance(warn_filtered_words, bool):
            raise TypeError(
                'warn_filtered_words must be a bool. Given: {}'.format(
                    warn_filtered_words))

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

    def __get_embeddings_from_word_set_(
            self,
            word_set: list,
            word_embedding: WordEmbeddingModel,
            warn_filtered_words: bool,
            lowercase_words: bool = False,
    ) -> dict:
        """Transforms a set of words into their respective embeddings and
        filters out words that are not in the model's vocabulary.

        Parameters
        ----------
        word_set : list
            The list/array with the word to be transformed.
        word_embedding : WordEmbeddingModel
            A word embedding pre-trained model.   
        warn_filtered_words : bool
            A flag that indicates if the function will warn about the filtered
            words.

        Returns
        -------
        dict
            A dict in which the keys are the remaining words and its values
            are the embeddings vectors.
        """

        # get the word embedding attributes
        embeddings = word_embedding.model_
        vocab_prefix = word_embedding.vocab_prefix_
        model_name = word_embedding.model_name_

        selected_embeddings = {}
        filtered_words = []

        # filter the words
        for word in word_set:
            # add the vocab prefix if is required.
            processed_word_lower = vocab_prefix + word.lower(
            ) if vocab_prefix != '' else word.lower()
            processed_word = vocab_prefix + word if vocab_prefix != '' else word

            # check if the word is in the word vector vocab
            if (processed_word in embeddings.vocab):
                # if it is, add the word vector to the return array
                selected_embeddings[processed_word] = embeddings[
                    processed_word]
            elif (processed_word_lower in embeddings.vocab):
                selected_embeddings[processed_word_lower] = embeddings[
                    processed_word_lower]

            else:
                filtered_words.append(processed_word)

        # warn if it is enabled
        if (warn_filtered_words and len(filtered_words) > 0):
            logging.warning(
                'The following words will not be considered because they '
                'do not exist in the Word Embedding ({}) vocabulary: {} '.
                format(model_name, filtered_words))

        return selected_embeddings

    def _get_embeddings_from_query(
            self,
            query: Query,
            word_embedding: WordEmbeddingModel,
            warn_filtered_words: bool = False,
            lost_vocabulary_threshold: float = 0.2
    ) -> Union[Tuple[List[dict], List[dict]], None]:
        """Obtains the word vectors associated with the provided Query.
        The words that does not appears in the word embedding pretrained model
        vocabulary are filtered.
        If the remaining words are percentage lower than the specified
        threshold, then the function will return none.

        Parameters
        ----------
        query : Query
            The query to be processed. From this, the words will be obtained
        word_embedding : WordEmbeddingModel
            A word embedding model.
        warn_filtered_words : bool, optional
            A flag that indicates if the function will print a warning with
            the filtered words (if any), by default False.

        Returns
        -------
        Union[Tuple[List[dict], List[dict]], None]
            Two lists with dictionaries that contains targets and attributes
            embeddings. Each dict key represents some word and its value
            represents its embedding vector. If any set has proportionally
            fewer words than the threshold, it returns None.
        """
        def is_percentage_of_filtered_words_under_threshold(
                embeddings, word_set, word_set_name, lost_words_threshold):
            remaining_words = list(embeddings.keys())
            number_of_filtered_words = len(word_set) - len(remaining_words)
            percentage_of_filtered_words = number_of_filtered_words / len(
                word_set)

            # if the percentage of filtered words are greater than the
            # threshold, log and return False
            if percentage_of_filtered_words > lost_words_threshold:
                logging.warning(
                    'Words lost during conversion of {} to {} embeddings '
                    'greater than the threshold of lost vocabulary ({} > {}).'.
                    format(
                        word_set_name if word_set_name != '' else
                        'Unnamed Word set', word_embedding.model_name_,
                        round(percentage_of_filtered_words,
                              2), lost_words_threshold))
                return True
            return False

        # check the inputs
        self._check_input(query, word_embedding, lost_vocabulary_threshold,
                          warn_filtered_words)

        some_set_has_fewer_words_than_the_threshold = False

        target_embeddings = []
        attribute_embeddings = []

        # get target sets embeddings
        for target_set, target_set_name in zip(query.target_sets_,
                                               query.target_sets_names_):
            embeddings = self.__get_embeddings_from_word_set_(
                target_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold,
            # log and change the flag.
            if is_percentage_of_filtered_words_under_threshold(
                    embeddings, target_set, target_set_name,
                    lost_vocabulary_threshold):
                some_set_has_fewer_words_than_the_threshold = True
            else:
                target_embeddings.append(embeddings)

        # get attribute sets embeddings
        for attribute_set, attribute_set_name in zip(
                query.attribute_sets_, query.attribute_sets_names_):
            embeddings = self.__get_embeddings_from_word_set_(
                attribute_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold,
            # log and change the flag.
            if is_percentage_of_filtered_words_under_threshold(
                    embeddings, attribute_set, attribute_set_name,
                    lost_vocabulary_threshold):
                some_set_has_fewer_words_than_the_threshold = True
            else:
                attribute_embeddings.append(embeddings)

        # check if some set has fewer words than the threshold. if that's
        #  the case, return None
        if some_set_has_fewer_words_than_the_threshold:
            logging.warning(
                'Some set in the query "{}" has fewer words than the allowed '
                'threshold. The processing of this query will return nan.'.
                format(query.query_name_))
            return None

        return target_embeddings, attribute_embeddings
