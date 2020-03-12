from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
import logging
import numpy as np
from typing import Union, NoReturn, Tuple


class BaseMetric:

    def __init__(self, metric_template: tuple, metric_name: str, metric_short_name: str):
        """Initializes a BaseMetric.

        Parameters
        ----------
        metric_template : tuple
            The template required by the metric.
        metric_name : str
            The name of the metric.
        metric_short_name : str
            The initials or short name of the metric.
        Raises
        ------
        TypeError
            If any element of the template is not an integer or string.
        """

        # check types of template_required
        if not isinstance(metric_template[0], (str, int)) or not isinstance(metric_template[1], (str, int)):
            raise TypeError(
                'Both components of template_required must be int or str. Given: {}'.format(metric_template))

        self.metric_template = metric_template
        self.metric_name = metric_name
        self.metric_short_name = metric_short_name

    def _check_input(self, query: Query, word_embedding: WordEmbeddingModel,
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
            A float number that indicates the acceptable threshold of lost words.
        warn_filtered_words : bool
            A bool that indicates if the word to embedding transformation will warn about the lost words.
        
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
            if the metric require different number of target sets than the delivered query
        Exception
            if the metric require different number of attribute sets than the delivered query
        """

        # check if the query passed is a instance of Query
        if not isinstance(query, Query):
            raise TypeError('query parameter must be a Query instance. Given {}'.format(query))

        # check if the word_embedding is a instance of WordEmbeddingModel
        if not isinstance(word_embedding, WordEmbeddingModel):
            raise TypeError('word_embedding must be a WordVectorsWrapper instance. Given: {}'.format(word_embedding))

        if not isinstance(lost_vocabulary_threshold, (float, np.float, np.float32, np.float64)):
            raise TypeError('lost_vocabulary_threshold must be a float. Given: {}'.format(lost_vocabulary_threshold))

        if not isinstance(warn_filtered_words, bool):
            raise TypeError('warn_filtered_words must be a bool. Given: {}'.format(warn_filtered_words))

        # check the cardinality of the target sets of the provided query
        if self.metric_template[0] != 'n' and query.template[0] != self.metric_template[0]:
            raise Exception(
                'The cardinality of the set of target words of the provided query ({}) is different from the cardinality required by {}: ({})'
                .format(query.template[0], self.metric_name, self.metric_template[0]))

        # check the cardinality of the attribute sets of the provided query
        if self.metric_template[1] != 'n' and query.template[1] != self.metric_template[1]:
            raise Exception(
                'The cardinality of the set of attributes words of the provided query ({}) is different from the cardinality required by {}: ({})'
                .format(query.template[1], self.metric_name, self.metric_template[1]))

    def __get_embeddings_from_word_set(self, word_set: list, word_embedding: WordEmbeddingModel,
                                       warn_filtered_words: bool) -> dict:
        """Transforms a set of words into their respective embeddings. Filters out words that are not in the model's vocabulary.
        
        Parameters
        ----------
        word_set : list
            The list/array with the word to be transformed.
        word_embedding : WordEmbeddingModel
            A word embedding pre-trained model .           
        warn_filtered_words : bool
            A flag that indicates if the function will warn about the filtered words.
        
        Returns
        -------
        dict
            A dict in which the keys are the remaining words and its values are the embeddings vectors. 
        """

        # get the word embedding attributes
        embeddings = word_embedding.word_embedding
        vocab_prefix = word_embedding.vocab_prefix
        model_name = word_embedding.model_name

        selected_embeddings = {}
        filtered_words = []

        # filter the words
        for word in word_set:
            # add the vocab prefix if is required.
            processed_word = vocab_prefix + word.lower() if vocab_prefix != '' else word.lower()

            # check if the word is in the word vector vocab
            if (processed_word in embeddings.vocab):
                # if it is, add the word vector to the return array
                selected_embeddings[processed_word] = embeddings[processed_word]
            else:
                filtered_words.append(processed_word)

        # warn if it is enabled
        if (warn_filtered_words and len(filtered_words) > 0):
            logging.warning(
                'The following words will not be considered because they do not exist in the Word Embedding ({}) vocabulary: {} '
                .format(model_name, filtered_words))

        return selected_embeddings

    def _get_embeddings_from_query(self, query: Query, word_embedding: WordEmbeddingModel,
                                   warn_filtered_words: bool = False, lost_words_threshold: float = 0.2
                                  ) -> Union[Tuple[np.ndarray, np.ndarray, list, list], None]:
        """Obtains the word vectors associated with the provided Query. 
        The words that does not appears in the word embedding pretrained model vocabulary are filtered. 
        If the remaining words are percentage lower than the specified threshold, then the function will return none.
        
        Parameters
        ----------
        query : Query
            The query to be processed. From this, the words will be obtained
        word_embedding : WordEmbeddingModel
            A word embedding model.
        warn_filtered_words : bool, optional
            A flag that indicates if the function will print a warning with the filtered words (if any), by default False.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray, list, list], None]
            4 iterables that contains in order: the target set embeddings, the attribute set embeddings, target set remaining words,  
            and the attribute set remaining words. If the remaining words are percentage lower than the specified threshold, 
            then the function will return only None.

        """

        def is_percentage_of_filtered_words_under_threshold(embeddings, word_set, word_set_name, lost_words_threshold):
            remaining_words = list(embeddings.keys())
            number_of_filtered_words = len(word_set) - len(remaining_words)
            percentage_of_filtered_words = number_of_filtered_words / len(word_set)

            # if the percentage of filtered words are greater than the threshold, log and return False
            if percentage_of_filtered_words > lost_words_threshold:
                logging.warning(
                    'Words lost during conversion of {} to {} embeddings greater than the threshold of lost vocabulary ({} > {}).'
                    .format(word_set_name if word_set_name != '' else 'Unnamed Word set', word_embedding.model_name,
                            round(percentage_of_filtered_words, 2), lost_words_threshold))
                return True
            return False

        some_set_has_fewer_words_than_the_threshold = False

        target_embeddings = []
        attribute_embeddings = []

        # get target sets embeddings
        for target_set, target_set_name in zip(query.target_sets, query.target_sets_names):
            embeddings = self.__get_embeddings_from_word_set(target_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold, log and change the flag.
            if is_percentage_of_filtered_words_under_threshold(embeddings, target_set, target_set_name,
                                                               lost_words_threshold):
                some_set_has_fewer_words_than_the_threshold = True
            else:
                target_embeddings.append(embeddings)

        # get attribute sets embeddings
        for attribute_set, attribute_set_name in zip(query.attribute_sets, query.attribute_sets_names):
            embeddings = self.__get_embeddings_from_word_set(attribute_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold, log and change the flag.
            if is_percentage_of_filtered_words_under_threshold(embeddings, attribute_set, attribute_set_name,
                                                               lost_words_threshold):
                some_set_has_fewer_words_than_the_threshold = True
            else:
                attribute_embeddings.append(embeddings)

        # check if some set has fewer words than the threshold. if that's the case, return None
        if some_set_has_fewer_words_than_the_threshold == True:
            logging.warning('Some set has fewer words than the allowed threshold. The metric will return nan.')
            return None

        return target_embeddings, attribute_embeddings

    def _generate_query_name(self, query: Query) -> str:
        """Generates the query name from the name of its target and attribute sets.
        
        Parameters
        ----------
        query : Query
            The query to be tested.
        
        Returns
        -------
        str
            The name of the query.
        """

        target_sets_names = query.target_sets_names
        attribute_sets_names = query.attribute_sets_names

        if len(target_sets_names) == 1:
            target = target_sets_names[0]
        if len(target_sets_names) == 2:
            target = target_sets_names[0] + " and " + target_sets_names[1]
        else:
            target = ', '.join([str(x) for x in target_sets_names]) + ' and ' + target_sets_names[-1]

        if len(attribute_sets_names) == 1:
            attribute = attribute_sets_names[0]
        if len(attribute_sets_names) == 2:
            attribute = attribute_sets_names[0] + " and " + attribute_sets_names[1]
        else:
            attribute = ', '.join([str(x) for x in attribute_sets_names]) + ' and ' + attribute_sets_names[-1]

        return target + ' wrt ' + attribute
