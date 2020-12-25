import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import gensim
import semantic_version
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

gensim_version = semantic_version.Version.coerce(gensim.__version__)

if gensim_version.major >= 4:
    from gensim.models import KeyedVectors as BaseKeyedVectors
else:
    from gensim.models.keyedvectors import BaseKeyedVectors

from .query import Query

EmbeddingDict = Dict[str, np.ndarray]
EmbeddingSets = Dict[str, EmbeddingDict]
PreprocessorArgs = Dict[str, Union[bool, str, Callable, None]]


class WordEmbeddingModel:
    """A container for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """
    def __init__(self,
                 model: BaseKeyedVectors,
                 model_name: str = None,
                 vocab_prefix: str = None):
        """Initializes the  container.

        Parameters
        ----------
        model : BaseKeyedVectors.
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

        >>> dummy_model = Word2Vec(common_texts, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.model_name)
        Dummy model dim=10
        >>> print(model.vocab_prefix)
        /en/


        Attributes
        ----------
        model : BaseKeyedVectors
            The model.
        vocab : 
            The vocabulary of the model (a dict with the words that have an associated 
            embedding in the model).
        model_name : str
            The name of the model.
        vocab_prefix : str
            A prefix that will be concatenated with each word of the vocab
            of the model.

        """

        # Type checking
        if not isinstance(model, BaseKeyedVectors):
            raise TypeError("model should be an instance of gensim\'s BaseKeyedVectors"
                            ", got {}.".format(model))

        if not isinstance(model_name, (str, type(None))):
            raise TypeError(
                'model_name should be a string or None, got {}.'.format(model_name))

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError(
                'vocab_prefix should be a string or None, got {}'.format(vocab_prefix))

        # Assign the attributes
        self.model = model

        # Obtain the vocabulary
        if gensim_version.major == 4:
            self.vocab = model.key_to_index
        else:
            self.vocab = model.vocab

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
        if key in self.vocab:
            return self.model[key]
        else:
            return None

    def _preprocess_word(
        self,
        word: str,
        preprocessor_args: PreprocessorArgs = {
            'strip_accents': False,
            'lowercase': False,
            'preprocessor': None,
        }
    ) -> str:
        """pre-processes a word before it is searched in the model's vocabulary.

        Parameters
        ----------
        word : str
            Word to be preprocessed.
        preprocessor_args : PreprocessorArgs, optional
            Dictionary with arguments that specifies how the words will be preprocessed,
            by default { 
                'strip_accents': False, 
                'lowercase': False, 
                'preprocessor': None, }

        Returns
        -------
        str
            The pre-processed word according to the given parameters.
        """

        preprocessor = preprocessor_args.get('preprocessor', None)
        if preprocessor and callable(preprocessor):
            word = preprocessor(word)

        else:
            if preprocessor_args.get('lowercase', False):
                word = word.lower()

            strip_accents = preprocessor_args.get('strip_accents', False)
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
        preprocessor_args: PreprocessorArgs = {},
        secondary_preprocessor_args: PreprocessorArgs = None,
    ) -> Tuple[List[str], EmbeddingDict]:
        """Transforms a set of words into their respective embeddings and
        discard out words that are not in the model's vocabulary (according to the rules 
        specified in the preprocessors).

        Parameters
        ----------
        word_set : List[str]
            The list/array with the words that will be transformed
        preprocessor_args : PreprocessorArgs, optional
            Dictionary with the arguments that specify how the pre-processing of the 
            words will be done, by default {}
            The options for the dict are: 
            - lowercase: bool. Indicates if the words are transformed to lowercase.
            - strip_accents: bool, {'ascii', 'unicode'}: Specifies if the accents of 
                             the words are eliminated. The stripping type can be 
                             specified. True uses 'unicode' by default.
            - preprocessor: Callable. It receives a function that operates on each 
                            word. In the case of specifying a function, it overrides 
                            the default preprocessor (i.e., the previous options 
                            stop working).
            
        secondary_preprocessor_args : PreprocessorArgs, optional
            Dictionary with arguments for pre-processing words (same as the previous 
            parameter), by default None.
            Indicates that in case a word is not found in the model's vocabulary 
            (using the default preprocessor or specified in preprocessor_args), 
            the function performs a second search for that word using the preprocessor 
            specified in this parameter.

        Returns
        -------
        Tuple[List[str], Dict[str, np.ndarray]]
            A tuple with a list of missing words and a dictionary that maps words 
            to embeddings.
        """

        if not isinstance(word_set, List):
            raise TypeError("word_set should be a list of strings"
                            ", got {}.".format(word_set))

        if not isinstance(preprocessor_args, Dict):
            raise TypeError("preprocessor_args should be a dict of preprocessor"
                            " arguments, got {}.".format(preprocessor_args))

        if not isinstance(secondary_preprocessor_args, (Dict, type(None))):
            raise TypeError("secondary_preprocessor_args should be a dict of "
                            "preprocessor arguments or None, got {}.".format(
                                secondary_preprocessor_args))

        selected_embeddings = {}
        not_found_words = []

        # filter the words
        for word in word_set:

            preprocessed_word = self._preprocess_word(word, preprocessor_args)
            embedding = self.__getitem__(preprocessed_word)

            if embedding is not None:
                selected_embeddings[preprocessed_word] = embedding

            # If the word was not found and also_search_for != {}
            # try the specified preprocessor configurations.
            elif secondary_preprocessor_args is not None:

                additional_preprocessed_word = self._preprocess_word(
                    word, secondary_preprocessor_args)

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
                "The following words from set '{}' do not exist within the vocabulary "
                "of {}: {}".format(set_name, self.model_name, not_found_words))

    def _check_lost_vocabulary_threshold(self, embeddings: EmbeddingDict,
                                         word_set: List[str], word_set_name: str,
                                         lost_words_threshold: float):

        remaining_words = list(embeddings.keys())
        number_of_lost_words = len(word_set) - len(remaining_words)
        percentage_of_lost_words = number_of_lost_words / len(word_set)

        # if the percentage of filtered words are greater than the
        # threshold, log and return False
        if percentage_of_lost_words > lost_words_threshold:
            logging.warning(
                "The transformation of '{}' into {} embeddings lost proportionally more "
                "words than specified in 'lost_words_threshold': {} lost with respect "
                "to {} maximum loss allowed.".format(word_set_name, self.model_name,
                                                     round(percentage_of_lost_words, 2),
                                                     lost_words_threshold))
            return True
        return False

    def get_embeddings_from_query(
        self,
        query: Query,
        lost_vocabulary_threshold: float = 0.2,
        preprocessor_args: PreprocessorArgs = {},
        secondary_preprocessor_args: PreprocessorArgs = None,
        warn_not_found_words: bool = False,
    ) -> Union[Tuple[EmbeddingSets, EmbeddingSets], None]:
        """Obtain the word vectors associated with the provided Query.

        The words that does not appears in the word embedding pretrained model
        vocabulary under the specified pre-processing are discarded.
        If the remaining words percentage in any query set is lower than the specified
        threshold, the function will return None.


        Parameters
        ----------
        query : Query
            The query to be processed.

        lost_vocabulary_threshold : float, optional, by default 0.2
            Indicates the proportional limit of words that any set of the query is 
            allowed to lose when transforming its words into embeddings. 
            In the case that any set of the query loses proportionally more words 
            than this limit, this method will return None.

        preprocessor_args : PreprocessorArgs, optional
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

        secondary_preprocessor_args : PreprocessorArgs, optional
            Dictionary with the arguments that specify how the secondary pre-processing 
            of the words will be done, by default None.
            Indicates that in case a word is not found in the model's vocabulary 
            (using the default preprocessor or specified in preprocessor_args), 
            the function performs a second search for that word using the preprocessor 
            specified in this parameter.
            Example: 
            Suppose we have the word "John" in the query and only the lowercase 
            version "john" is found in the model's vocabulary.
            If we use preprocessor_args by default (so as not to affect the search 
            for other words that may exist in capital letters in the model), the 
            function will not be able to extract the representation of "john" even 
            if it exists in lower case.
            However, we can use {'lowecase' : True} in preprocessor_args 
            to specify that it also looks for the lower case version of "juan", 
            without affecting the first preprocessor. Thus, this preprocessor will 
            only remain as an alternative in case the first one does not work.
            
        warn_not_found_words : bool, optional
            A flag that indicates if the function will warn (in the logger)
            the words that were not found in the model's vocabulary, 
            by default False.

        Returns
        -------
        Union[Tuple[EmbeddingSets, EmbeddingSets], None]
            A tuple of dictionaries containing the targets and attribute sets or None 
            in case there is a set that has proportionally less embeddings than it was 
            allowed to lose.

        Raises
        ------
        TypeError
            If query is not an instance of Query
        TypeError
            If lost_vocabulary_threshold is not float
        TypeError
            If preprocessor_args is not a dictionary
        TypeError
            If secondary_preprocessor_args is not a dictionary
        TypeError
            If warn_not_found_words is not a boolean
        """
        # Type check
        if not isinstance(query, Query):
            raise TypeError(
                "query should be an instance of Query, got {}.".format(query))

        if not isinstance(lost_vocabulary_threshold, (float, (np.floating, float))):
            raise TypeError(
                "lost_vocabulary_threshold should be float or np.floating, got {}.".
                format(lost_vocabulary_threshold))

        if not isinstance(preprocessor_args, Dict):
            raise TypeError("preprocessor_args should be a dict of preprocessor"
                            " arguments, got {}.".format(preprocessor_args))

        if not isinstance(secondary_preprocessor_args, (Dict, type(None))):
            raise TypeError("secondary_preprocessor_args should be a dict of "
                            "preprocessor arguments or None, got {}.".format(
                                secondary_preprocessor_args))

        if not isinstance(warn_not_found_words, bool):
            raise TypeError("warn_not_found_words should be a boolean, got {}.".format(
                warn_not_found_words))

        some_set_lost_more_words_than_threshold: bool = False

        target_embeddings: EmbeddingDict = {}
        attribute_embeddings: EmbeddingDict = {}

        # --------------------------------------------------------------------
        # get target sets embeddings
        # --------------------------------------------------------------------
        for target_set, target_set_name in zip(query.target_sets,
                                               query.target_sets_names):
            not_found_words, obtained_embeddings = self.get_embeddings_from_word_set(
                target_set, preprocessor_args, secondary_preprocessor_args)

            # warn not found words if it is enabled.
            if warn_not_found_words:
                self._warn_not_found_words(target_set_name, not_found_words)

            # if the lost words are greater than the threshold,
            # warn and change the flag.
            if self._check_lost_vocabulary_threshold(obtained_embeddings, target_set,
                                                     target_set_name,
                                                     lost_vocabulary_threshold):
                some_set_lost_more_words_than_threshold = True

            target_embeddings[target_set_name] = obtained_embeddings

        # --------------------------------------------------------------------
        # get attribute sets embeddings
        # --------------------------------------------------------------------
        for attribute_set, attribute_set_name in zip(query.attribute_sets,
                                                     query.attribute_sets_names):

            not_found_words, obtained_embeddings = self.get_embeddings_from_word_set(
                attribute_set, preprocessor_args, secondary_preprocessor_args)

            if warn_not_found_words:
                # warn not found words if it is enabled.
                self._warn_not_found_words(attribute_set_name, not_found_words)

            # if the filtered words are greater than the threshold,
            # log and change the flag.
            if self._check_lost_vocabulary_threshold(obtained_embeddings, attribute_set,
                                                     attribute_set_name,
                                                     lost_vocabulary_threshold):
                some_set_lost_more_words_than_threshold = True

            attribute_embeddings[attribute_set_name] = obtained_embeddings

        # check if some set has fewer words than the threshold. if that's
        #  the case, return None
        if some_set_lost_more_words_than_threshold:
            logging.error(
                "At least one set of '{}' query has proportionally fewer embeddings "
                "than allowed by the lost_vocabulary_threshold parameter ({}). "
                "This query will return np.nan.".format(query.query_name,
                                                        lost_vocabulary_threshold))
            return None

        return target_embeddings, attribute_embeddings
