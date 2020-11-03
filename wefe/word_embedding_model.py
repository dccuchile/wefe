from typing import Callable, Dict, Iterable, List, Tuple, Union
from gensim.models.keyedvectors import BaseKeyedVectors
import logging
import string
import unidecode
import numpy as np

from .query import Query


# TODO: Incluir pydantic
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

        if not isinstance(word_embedding, BaseKeyedVectors):
            raise TypeError('word_embedding must be an instance of a gensim\'s'
                            ' KeyedVectors. Given: {}'.format(word_embedding))
        else:
            self.model_ = word_embedding

        if model_name is None:
            self.model_name_ = 'Unnamed word embedding model'
        elif not isinstance(model_name, str):
            raise TypeError(
                'model_name must be a string. Given: {}'.format(model_name))
        else:
            self.model_name_ = model_name

        if vocab_prefix is None:
            self.vocab_prefix_ = ''
        elif not isinstance(vocab_prefix, str):
            raise TypeError(
                'vocab_prefix parameter must be a string. Given: {}'.format(
                    vocab_prefix))
        else:
            self.vocab_prefix_ = vocab_prefix

    def __eq__(self, other):
        if self.model_ != other.model_:
            return False
        if self.model_name_ != other.model_name_:
            return False
        return True

    # TODO: Implementar get_embedding que actue según el tipo de embedding cargado

    def _get_embedding(self, word: str) -> Union[np.ndarray, None]:
        if word in self.model_.vocab:
            return self.model_[word]
        return None

    def _preprocess_word(
            self, word: str,
            preprocessor_options: Dict[str, Union[bool, str,
                                                  Callable]]) -> str:

        preprocessor = preprocessor_options.get('custom_preprocesor', False)
        if preprocessor and callable(preprocessor):
            word = preprocessor(word)

        else:
            if preprocessor_options.get('lowercase', False):
                word = word.lower()

            if preprocessor_options.get('remove_punctuations', False):
                word = word.translate(self.punctuation_translator_)

            if preprocessor_options.get('translate_to_ascii', False):
                word = unidecode.unidecode(word)

        if self.vocab_prefix_ is not None and self.vocab_prefix_ != '':
            word = self.vocab_prefix_ + word
        return word

    def __get_embeddings_from_word_set_(
            self, word_set: Iterable,
            word_preprocessor_options: Dict[str, [bool, str, Callable]] = {},
            also_search_for: Dict[str, [bool, str, Callable]] = {},
            warn_not_found_words: bool = False) -> dict:
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

            preprocessed_word = self._preprocess_word(
                word, word_preprocessor_options)
            embedding = self._get_embedding(preprocessed_word)

            if embedding != None:
                selected_embeddings[preprocessed_word] = embedding

            # If the word was not found, try the also search for # preprocessor 
            # configurations
            else:
                preprocessed_word_alt = self._preprocess_word(
                    word, also_search_for)
                embedding = self._get_embedding(preprocessed_word_alt)

                if embedding != None:
                    selected_embeddings[
                        preprocessed_word_alt] = embedding
                else:
                    not_found_words.append(preprocessed_word)
            else:
                not_found_words.append(preprocessed_word)

        # warn the filtered words if it is enabled
        if (warn_not_found_words and len(not_found_words) > 0):
            logging.warning(
                'The following words will not be considered because they '
                'do not exist in the Word Embedding ({}) vocabulary: {} '.
                format(self.model_name_, not_found_words))

        return selected_embeddings

    def _get_embeddings_from_query(
        self,
        query: Query,
        lost_vocabulary_threshold: float = 0.2,
        word_preprocessor_options: Dict[str, [bool, str, Callable]] = {},
        also_search_for: Dict[str, [bool, str, Callable]] = {},
        warn_filtered_words: bool = False,
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
                        'Unnamed Word set', self.model_name_,
                        round(percentage_of_filtered_words,
                              2), lost_words_threshold))
                return True
            return False

        some_set_has_fewer_words_than_the_threshold = False

        target_embeddings = []
        attribute_embeddings = []

        # get target sets embeddings
        for target_set, target_set_name in zip(query.target_sets_,
                                               query.target_sets_names_):
            embeddings = self.__get_embeddings_from_word_set_(
                target_set, warn_filtered_words, word_preprocessor_options,
                also_search_for_lowecase)

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
                attribute_set, warn_filtered_words, word_preprocessor_options,
                also_search_for_lowecase)

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
