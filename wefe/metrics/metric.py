from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
import logging
import numpy as np


class Metric:

    def __init__(self, metric_template: tuple, metric_name: str, metric_short_name: str):
        self.metric_template = metric_template
        self.metric_name = metric_name
        self.metric_short_name = metric_short_name

    def check_input(self, query: Query, word_embedding: WordEmbeddingModel):
        """Checks if the input of a metric is valid.
        
        Parameters
        ----------
        query : Query
            The query that the method will execute.
        word_embedding : WordEmbeddingModel
            A word embedding model.
        
        Raises
        ------
        TypeError
            if query is not instance of Query.
        TypeError
            if word_embedding is not instance of WordEmbeddingModel.
        TypeError
            if one or both elements of the template are not strings or int.
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
            raise TypeError('word_embedding must be a WordVectorsWrapper instance. Given: {}'.format(
                type(word_embedding)))

        # check types of template_required
        if not isinstance(self.metric_template[0], (str, int)) or not isinstance(self.metric_template[1], (str, int)):
            raise TypeError('Both components of template_required must be int or str. Given: {}'.format(
                self.metric_template))

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
                                       warn_filtered_words: bool) -> tuple:
        """[summary]
        
        Parameters
        ----------
        word_set : list
            [description]
        word_embedding : WordEmbeddingModel
            [description]
        vocab_prefix : str
            [description]
        warn_filtered_words : bool
            [description]
        
        Returns
        -------
        tuple
            A tuple with a numpy array that contains the obtained word embeddings and a list with the preserved words 
            (which were not filtered out because they were not in the vocabulary).
        """

        # get the word embedding attributes
        word_vectors = word_embedding.word_embedding
        vocab_prefix = word_embedding.vocab_prefix
        model_name = word_embedding.model_name

        embeddings = []
        preserved_words = []
        filtered_words = []

        # filter the words
        for word in word_set:
            # add the vocab prefix if is required.
            processed_word = vocab_prefix + word.lower() if vocab_prefix != '' else word.lower()

            # check if the word is in the word vector vocab
            if (processed_word in word_vectors.vocab):
                # if it is, add the word vector to the return array
                embeddings.append(word_vectors[processed_word])
                preserved_words.append(processed_word)
            else:
                filtered_words.append(processed_word)

        # warn if it is enabled
        if (warn_filtered_words and len(filtered_words) > 0):
            logging.warning(
                'The following words will not be considered because they do not exist in the Word Embedding ({}) vocabulary: {} '
                .format(model_name, filtered_words))

        return np.array(embeddings), preserved_words, filtered_words

    def get_embeddings_from_query(self, query: Query, word_embedding: WordEmbeddingModel,
                                  warn_filtered_words: bool = False, lost_words_threshold: float = 0.2) -> tuple:
        """Obtains the word vectors associated with the provided Query. 
        The words that does not appears in the word vector model vocabulary are filtered.
        
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
        tuple
            4 arrays that contains in order: the target set embeddings, target set remaining words, the attribute set embeddings 
            and the attribute set remaining words.

        Raises
        ------
        TypeError
            if word_embedding is not a WordEmbeddingModel instance.
        TypeError
            if warn_filtered_words is not a boolean.
        """

        if not isinstance(word_embedding, WordEmbeddingModel):
            raise TypeError(
                'word_embedding must be a instance of WordEmbeddingModel class. Given: {}'.format(WordEmbeddingModel))

        if not isinstance(query, Query):
            raise TypeError('query parameter must be a Query instance. Given {}'.format(query))

        if not isinstance(warn_filtered_words, bool):
            raise TypeError('warn_filtered_words must be a boolean. Given: {}'.format(warn_filtered_words))

        some_set_has_fewer_words_than_the_threshold = False

        target_embeddings = []
        target_preserved_words = []
        attribute_embeddings = []
        attribute_preserved_words = []

        # get target set embeddings
        for target_set, target_set_name in zip(query.target_sets, query.target_sets_names):
            embeddings, remaining_words, filtered_words = self.__get_embeddings_from_word_set(
                target_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold, log and change the flag to return none.
            if (len(filtered_words) / len(target_set) > lost_words_threshold):
                logging.warning(
                    'Words lost during conversion of {} to {} embeddings greater than the threshold of lost vocabulary ({} > {}).'
                    .format('of ' + target_set_name if target_set_name != '' else '', word_embedding.model_name,
                            round(len(filtered_words) / len(target_set), 2), lost_words_threshold))
                some_set_has_fewer_words_than_the_threshold = True

            else:
                target_embeddings.append(embeddings)
                target_preserved_words.append(remaining_words)

        # get attribute set embeddings and remaining words
        for attribute_set, attribute_set_name in zip(query.attribute_sets, query.attribute_sets_names):
            embeddings, remaining_words, filtered_words = self.__get_embeddings_from_word_set(
                attribute_set, word_embedding, warn_filtered_words)

            # if the filtered words are greater than the threshold, log and change the flag to return none
            if (len(filtered_words) / len(attribute_set) > lost_words_threshold):
                logging.warning(
                    'Words lost during conversion of {} to {} embeddings greater than the threshold of lost vocabulary ({} > {}).'
                    .format('of ' + attribute_set_name if attribute_set_name != '' else '', word_embedding.model_name,
                            round(len(filtered_words) / len(attribute_set), 2), lost_words_threshold))
                some_set_has_fewer_words_than_the_threshold = True

            else:
                attribute_embeddings.append(embeddings)
                attribute_preserved_words.append(remaining_words)

        if some_set_has_fewer_words_than_the_threshold == True:
            logging.warning('Some set has fewer words than the allowed threshold. The metric will return nan.')
            return None

        return target_embeddings, attribute_embeddings, target_preserved_words, attribute_preserved_words

    def generate_query_name(self, query: Query):

        target_sets_names = query.target_sets_names
        attribute_sets_names = query.attribute_sets_names

        if len(target_sets_names) == 1:
            target = target_sets_names[0]
        if len(target_sets_names) == 2:
            target = target_sets_names[0] + " and " + target_sets_names[1]
        else:
            target = target_sets_names[0:-1].join(' ,') + ' and ' + target_sets_names[-1]

        if len(attribute_sets_names) == 1:
            attribute = attribute_sets_names[0]
        if len(attribute_sets_names) == 2:
            attribute = attribute_sets_names[0] + " and " + attribute_sets_names[1]
        else:
            attribute = attribute_sets_names[0:-1].join(' ,') + ' and ' + attribute_sets_names[-1]

        return target + ' wrt ' + attribute
