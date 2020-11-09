from typing import Callable, Dict, Iterable, List, NoReturn, Tuple, Union
from gensim.models.keyedvectors import BaseKeyedVectors
import logging
import string
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

import numpy as np

from .query import Query


# TODO: Incluir otras fuentes de embeddings. (como polyglot o leer w2v)
class WordEmbeddingModel:
    """A container for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """

    punctuation_translator_ = str.maketrans('', '', string.punctuation)

    def __init__(self,
                 word_embedding: BaseKeyedVectors,
                 model_name: str = None,
                 vocab_prefix: str = None):
        """Initializes the WordEmbeddingModel container.

        Parameters
        ----------
        keyed_vectors : BaseKeyedVectors.
            An instance of word embedding loaded through gensim KeyedVector
            interface or gensim's api.
        model_name : str, optional
            The name of the model, by default ''.
        vocab_prefix : str, optional.
            A prefix that will be concatenated with all word in the model
            vocab, by default None.

        Raises
        ------
        TypeError
            if word_embedding is not a KeyedVectors instance.
        TypeError
            if model_name is not None and not instance of str.
        TypeError
            if vocab_prefix is not None and not instance of str.

        Examples
        --------
        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from wefe.word_embedding_model import WordEmbeddingModel

        >>> dummy_model = Word2Vec(common_texts, size=10, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.model_name_)
        Dummy model dim=10
        >>> print(model.vocab_prefix_)
        /en/


        Attributes
        ----------
        model_ : KeyedVectors
            The object that contains the model.
        model_name_ : str
            The name of the model.
        vocab_prefix_ : str
            A prefix that will be concatenated with each word of the vocab
            of the model.

        """

        # Type checking
        if not isinstance(word_embedding, BaseKeyedVectors):
            raise TypeError(
                "word_embedding should be an instance of gensim\'s BaseKeyedVectors, got {}."
                .format(word_embedding))

        if not isinstance(model_name, (str, type(None))):
            raise TypeError(
                'model_name should be a string or None, got {}.'.format(model_name))

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError(
                'vocab_prefix should be a string or None, got {}'.format(vocab_prefix))

        # Assign
        self.model = word_embedding
        self.vocab_prefix = vocab_prefix
        if model_name is None:
            self.model_name = 'Unnamed word embedding model'
        else:
            self.model_name = model_name

    def __eq__(self, other):
        if self.model != other.model:
            return False
        if self.model_name != other.model_name:
            return False
        if self.vocab_prefix != other.vocab_prefix:
            return False
        return True

    def __getitem__(self, key):
        if key in self.model.vocab:
            return self.model[key]
        return None

    def _preprocess_word(
            self,
            word: str,
            preprocessor_options: Dict[str, Union[bool, str, Callable]] = {}) -> str:

        preprocessor = preprocessor_options.get('preprocessor', None)
        if preprocessor and callable(preprocessor):
            word = preprocessor(word)

        else:
            if preprocessor_options.get('lowercase', False):
                word = word.lower()

            if preprocessor_options.get('strip_punctuation', False):
                word = word.translate(self.punctuation_translator_)

            strip_accents = preprocessor_options.get('strip_accents', False)
            if strip_accents == True:
                word = strip_accents_unicode(word)
            elif strip_accents == 'ascii':
                word = strip_accents_ascii(word)
            elif strip_accents == 'unicode':
                word = strip_accents_unicode(word)

        if self.vocab_prefix is not None:
            word = self.vocab_prefix + word

        return word

    def get_embeddings_from_word_set(
        self,
        word_set: List[str],
        preprocessor_options: Dict[str, Union[bool, str, Callable]] = {},
        also_search_for: Dict[str, Union[bool, str, Callable]] = {},
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Transforms a set of words into their respective embeddings and
        filters out words that are not in the model's vocabulary.

        Parameters
        ----------
        word_set : list
            The list/array with the word to be transformed.
        warn_filtered_words : bool
            A flag that indicates if the function will warn about the filtered
            words.

        Returns
        -------
        dict
            A dict in which the keys are the remaining words and its values
            are the embeddings vectors.
        """

        selected_embeddings = {}
        not_found_words = []

        # filter the words
        for word in word_set:

            preprocessed_word = self._preprocess_word(word, preprocessor_options)
            embedding = self.__getitem__(preprocessed_word)

            if embedding is not None:
                selected_embeddings[preprocessed_word] = embedding

            # If the word was not found and also_search_for != {}
            # try the specified preprocessor configurations.
            elif also_search_for != {}:

                additional_preprocessed_word = self._preprocess_word(
                    word, also_search_for)

                embedding = self.__getitem__(additional_preprocessed_word)

                if embedding is not None:
                    selected_embeddings[additional_preprocessed_word] = embedding
                else:
                    not_found_words.append(word)

            # if also_search_for == {}, just add the word to not_found_words.
            else:
                not_found_words.append(word)

        return not_found_words, selected_embeddings

    def _warn_not_found_words(self, set_name: str, not_found_words: List[str]) -> None:

        if len(not_found_words) > 0:
            logging.warning(
                "The following words from set '{}' do not exist within the vocabulary"
                "of {}: {}".format(set_name, self.model_name, not_found_words))

    def _check_lost_vocabulary_threshold(self, embeddings, word_set, word_set_name,
                                         lost_words_threshold):
        remaining_words = list(embeddings.keys())
        number_of_lost_words = len(word_set) - len(remaining_words)
        percentage_of_lost_words = number_of_lost_words / len(word_set)

        # if the percentage of filtered words are greater than the
        # threshold, log and return False
        if percentage_of_lost_words > lost_words_threshold:
            logging.warning(
                "The transformation of '{}' into {} embeddings lost proportionally more"
                "words than specified in 'lost_words_threshold': {} lost with respect to"
                "{} maximum loss allowed.".format(word_set_name, self.model_name,
                                                  round(percentage_of_lost_words, 2),
                                                  lost_words_threshold))
            return True
        return False

    def get_embeddings_from_query(
        self,
        query: Query,
        lost_vocabulary_threshold: float = 0.2,
        preprocessor_options: Dict[str, Union[bool, str, Callable]] = {},
        also_search_for: Dict[str, Union[bool, str, Callable]] = {},
        warn_not_found_words: bool = False,
    ) -> Union[Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]], None]:
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
        warn_not_found_words : bool, optional
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

        # Type check
        if not isinstance(query, Query):
            raise TypeError(
                "query should be an instance of Query, got {}.".format(query))

        if not isinstance(lost_vocabulary_threshold, (float, (np.floating, float))):
            raise TypeError(
                "lost_vocabulary_threshold should be float or np.floating, got {}.".
                format(query))

        if not isinstance(preprocessor_options, Dict):
            raise TypeError(
                "word_preprocessor_options should be a dict of preprocessor options, got {}."
                .format(query))

        if not isinstance(also_search_for, Dict):
            raise TypeError(
                "also_search_for should be a dict of preprocessor options, got {}.".
                format(query))

        if not isinstance(warn_not_found_words, bool):
            raise TypeError(
                "warn_not_found_words should be a boolean, got {}.".format(query))

        some_set_lost_more_words_than_threshold: bool = False

        target_embeddings = []
        attribute_embeddings = []

        # --------------------------------------------------------------------
        # get target sets embeddings
        # --------------------------------------------------------------------
        for target_set, target_set_name in zip(query.target_sets_,
                                               query.target_sets_names_):
            not_found_words, embeddings = self.get_embeddings_from_word_set(
                target_set, preprocessor_options, also_search_for)

            # warn not found words if it is enabled.
            if warn_not_found_words:
                self._warn_not_found_words(target_set_name, not_found_words)

            # if the lost words are greater than the threshold,
            # warn and change the flag.
            if self._check_lost_vocabulary_threshold(embeddings, target_set,
                                                     target_set_name,
                                                     lost_vocabulary_threshold):
                some_set_lost_more_words_than_threshold = True

            target_embeddings.append(embeddings)

        # --------------------------------------------------------------------
        # get attribute sets embeddings
        # --------------------------------------------------------------------
        for attribute_set, attribute_set_name in zip(query.attribute_sets_,
                                                     query.attribute_sets_names_):

            not_found_words, embeddings = self.get_embeddings_from_word_set(
                attribute_set, preprocessor_options, also_search_for)

            if warn_not_found_words:
                # warn not found words if it is enabled.
                self._warn_not_found_words(attribute_set, not_found_words)

            # if the filtered words are greater than the threshold,
            # log and change the flag.
            if self._check_lost_vocabulary_threshold(embeddings, attribute_set,
                                                     attribute_set_name,
                                                     lost_vocabulary_threshold):
                some_set_lost_more_words_than_threshold = True

            attribute_embeddings.append(embeddings)

        # check if some set has fewer words than the threshold. if that's
        #  the case, return None
        if some_set_lost_more_words_than_threshold:
            logging.error(
                "At least one set of '{}' query has proportionally fewer embeddings"
                "than allowed by the lost_vocabulary_threshold parameter ({})."
                "This query will return null.".format(query.query_name_,
                                                      lost_vocabulary_threshold))
            return None

        return target_embeddings, attribute_embeddings
