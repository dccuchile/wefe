"""Module with utilities that ease the transformation of word sets to embeddings."""
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

EmbeddingDict = Dict[str, np.ndarray]
EmbeddingSets = Dict[str, EmbeddingDict]


def preprocess_word(
    word: str,
    options: Dict[str, Union[str, bool, Callable]] = {},
    vocab_prefix: Optional[str] = None,
) -> str:
    """pre-processes a word before it is searched in the model's vocabulary.

    Parameters
    ----------
    word : str
        Word to be preprocessed.
    options : Dict[str, Union[str, bool, Callable]], optional
        Dictionary with arguments that specifies how the words will be preprocessed,
        The available word preprocessing options are as follows:

        - ```lowercase```: bool. Indicates if the words are transformed to lowercase.

        - ```uppercase```: bool. Indicates if the words are transformed to uppercase.

        - ```titlecase```: bool. Indicates if the words are transformed to titlecase.

        - ```strip_accents```: `bool`, `{'ascii', 'unicode'}`: Specifies if the accents of
          the words are eliminated. The stripping type can be
          specified. True uses 'unicode' by default.

        - ```preprocessor```: Callable. It receives a function that operates on each
          word. In the case of specifying a function, it overrides
          the default preprocessor (i.e., the previous options
          stop working).

        By default, no preprocessing is generated, which is equivalent to {}

    Returns
    -------
    str
        The pre-processed word according to the given parameters.
    """
    preprocessor = options.get("preprocessor", None)

    # if the preprocessor is specified, it takes precedence over all other operations.
    if preprocessor is not None and callable(preprocessor):
        word = preprocessor(word)

    else:
        strip_accents = options.get("strip_accents", False)
        lowercase = options.get("lowercase", False)
        uppercase = options.get("uppercase", False)
        titlecase = options.get("titlecase", False)

        if lowercase:
            word = word.lower()
        elif uppercase:
            word = word.upper()
        elif titlecase:
            word = word.title()

        # by default, if strip_accents is True, run strip_accents_unicode
        if strip_accents is True:
            word = strip_accents_unicode(word)
        elif strip_accents == "ascii":
            word = strip_accents_ascii(word)
        elif strip_accents == "unicode":
            word = strip_accents_unicode(word)

    if vocab_prefix is not None and isinstance(vocab_prefix, str):
        return vocab_prefix + word
    return word


def get_embeddings_from_set(
    model: WordEmbeddingModel,
    word_set: Sequence[str],
    preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
    strategy: str = "first",
    normalize: bool = False,
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Transform a sequence of words into dictionary that maps word - word embedding.

    The method discard out words that are not in the model's vocabulary
    (according to the rules specified in the preprocessors).


    Parameters
    ----------
    model : WordEmbeddingModel
        A word embeddding model

    word_set : Sequence[str]
        A sequence with the words that this function will convert to embeddings.

    preprocessors : List[Dict[str, Union[str, bool, Callable]]]
        A list with preprocessor options.

        A ``preprocessor`` is a dictionary that specifies what processing(s) are
        performed on each word before it is looked up in the model vocabulary.
        For example, the ``preprocessor``
        ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
        and remove the accent from each word before searching for them in the
        model vocabulary. Note that an empty dictionary ``{}`` indicates that no
        preprocessing is done.

        The possible options for a preprocessor are:

        *   ``lowercase``: ``bool``. Indicates that the words are transformed to
            lowercase.
        *   ``uppercase``: ``bool``. Indicates that the words are transformed to
            uppercase.
        *   ``titlecase``: ``bool``. Indicates that the words are transformed to
            titlecase.
        *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
            the accents of the words are eliminated. The stripping type can be
            specified. True uses ‘unicode’ by default.
        *   ``preprocessor``: ``Callable``. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        ``[{}, {"lowercase": True, "strip_accents": True}]``
        ``{}`` allows first to search for the original words in the vocabulary of
        the model. In case some of them are not found,
        ``{"lowercase": True, "strip_accents": True}`` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : str, optional
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. all' will include all
        transformed words found, by default "first".

    normalize : bool, optional
        True indicates that embeddings will be normalized, by default False

    verbose : bool, optional
        Indicates whether the execution status of this function is printed,
        by default False

    Returns
    -------
    Tuple[List[str], Dict[str, np.ndarray]]
        A tuple containing the words that could not be found and a dictionary with
        the found words and their corresponding embeddings.
    """
    # ----------------------------------------------------------------------------------
    # type verifications.

    if not isinstance(model, WordEmbeddingModel):
        raise TypeError(f"model should be a WordEmbeddingModel instance, got {model}.")

    if not isinstance(word_set, (list, tuple, np.ndarray)):
        raise TypeError(
            "word_set should be a list, tuple or np.array of strings"
            f", got {word_set}."
        )

    if not isinstance(preprocessors, list):
        raise TypeError(
            "preprocessors should be a list of dicts which contains preprocessor options"
            f", got {preprocessors}."
        )
    if len(preprocessors) == 0:
        raise TypeError(
            "preprocessors must indicate at least one preprocessor, even if it is "
            "an empty dictionary {}, "
            f"got: {preprocessors}."
        )
    for idx, p in enumerate(preprocessors):
        if not isinstance(p, dict):
            raise TypeError(
                f"each preprocessor should be a dict, got {p} at index {idx}."
            )

    if strategy != "first" and strategy != "all":
        raise ValueError(f"strategy should be 'first' or 'all', got {strategy}.")

    # ----------------------------------------------------------------------------------
    # filter the words

    selected_embeddings = {}
    not_found_words = []

    for word in word_set:

        for preprocessor in preprocessors:
            preprocessed_word = preprocess_word(
                word, options=preprocessor, vocab_prefix=model.vocab_prefix
            )
            embedding = model[preprocessed_word]

            if embedding is not None:
                selected_embeddings[preprocessed_word] = embedding

                # if the selected strategy is first, then it stops on the first
                # word encountered.
                if strategy == "first":
                    break
            else:
                not_found_words.append(preprocessed_word)

    # if requested, normalize embeddings.
    if normalize:
        selected_embeddings = {
            k: v / np.linalg.norm(v) for k, v in selected_embeddings.items()
        }

    if verbose:
        print(
            f"Word(s) found: {list(selected_embeddings.keys())}, "
            f"not found: {not_found_words}"
        )

    return not_found_words, selected_embeddings


def _warn_not_found_words(
    warn_not_found_words: bool,
    not_found_words: List[str],
    model_name: str,
    set_name: str,
) -> None:

    if not isinstance(warn_not_found_words, bool):
        raise TypeError(
            "warn_not_found_words should be a boolean, got {}.".format(
                warn_not_found_words
            )
        )

    if warn_not_found_words:

        if len(not_found_words) > 0:
            logging.warning(
                "The following words from set '{}' do not exist within the vocabulary "
                "of {}: {}".format(set_name, model_name, not_found_words)
            )


def _check_lost_vocabulary_threshold(
    model: WordEmbeddingModel,
    embeddings: EmbeddingDict,
    word_set: List[str],
    word_set_name: str,
    lost_vocabulary_threshold: float,
):

    if not isinstance(lost_vocabulary_threshold, (float, np.floating)):
        raise TypeError(
            "lost_vocabulary_threshold should be float, "
            "got {}.".format(lost_vocabulary_threshold)
        )

    remaining_words = list(embeddings.keys())
    number_of_lost_words = len(word_set) - len(remaining_words)
    percentage_of_lost_words = number_of_lost_words / len(word_set)

    # if the percentage of filtered words are greater than the
    # threshold, log and return False
    if percentage_of_lost_words > lost_vocabulary_threshold:
        logging.warning(
            "The transformation of '{}' into {} embeddings lost proportionally more "
            "words than specified in 'lost_words_threshold': {} lost with respect "
            "to {} maximum loss allowed.".format(
                word_set_name,
                model.name,
                round(percentage_of_lost_words, 2),
                lost_vocabulary_threshold,
            )
        )
        return True
    return False


def get_embeddings_from_sets(
    model: WordEmbeddingModel,
    sets: Sequence[Sequence[str]],
    sets_name: Union[str, None] = None,
    preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
    strategy: str = "first",
    normalize: bool = False,
    discard_incomplete_sets: bool = True,
    warn_lost_sets: bool = True,
    verbose: bool = False,
) -> List[EmbeddingDict]:
    """Given a sequence of word sets, obtain their corresponding embeddings.

    Parameters
    ----------
    model

    sets :  Sequence[Sequence[str]]
        A sequence containing word sets.
        Example: `[['woman', 'man'], ['she', 'he'], ['mother', 'father'] ...]`.

    sets_name : Union[str, optional]
        The name of the set of word sets. Example: `definning sets`.
        This parameter is used only for printing.
        by default None

    preprocessors : List[Dict[str, Union[str, bool, Callable]]]
        A list with preprocessor options.

        A ``preprocessor`` is a dictionary that specifies what processing(s) are
        performed on each word before it is looked up in the model vocabulary.
        For example, the ``preprocessor``
        ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
        and remove the accent from each word before searching for them in the
        model vocabulary. Note that an empty dictionary ``{}`` indicates that no
        preprocessing is done.

        The possible options for a preprocessor are:

        *   ``lowercase``: ``bool``. Indicates that the words are transformed to
            lowercase.
        *   ``uppercase``: ``bool``. Indicates that the words are transformed to
            uppercase.
        *   ``titlecase``: ``bool``. Indicates that the words are transformed to
            titlecase.
        *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
            the accents of the words are eliminated. The stripping type can be
            specified. True uses ‘unicode’ by default.
        *   ``preprocessor``: ``Callable``. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        ``[{}, {"lowercase": True, "strip_accents": True}]``
        ``{}`` allows first to search for the original words in the vocabulary of
        the model. In case some of them are not found,
        ``{"lowercase": True, "strip_accents": True}`` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : str, optional
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. all' will include all
        transformed words found, by default "first".

    normalize : bool, optional
        True indicates that embeddings will be normalized, by default False

    discard_incomplete_sets : bool, optional
        True indicates that if a set could not be completely converted, it will be
        discarded., by default True

    warn_lost_sets : bool, optional
        Indicates whether word sets that cannot be fully converted to embeddings
        are warned in the logger,
        by default True

    verbose : bool, optional
        Indicates whether the execution status of this function is printed,
        by default False

    Returns
    -------
    List[EmbeddingDict]
        A list of dictionaries. Each dictionary contains as keys a pair of words
        and as values their associated embeddings.
    """
    if not isinstance(sets, (list, tuple, np.ndarray)):
        raise TypeError(
            "sets should be a sequence of sequences (list, tuple or np.array) "
            f"of strings, got: {type(sets)}."
        )

    for idx, set_ in enumerate(sets):
        if not isinstance(set_, (list, tuple, np.ndarray)):
            raise TypeError(
                "Every set in sets should be a list, tuple or np.array of "
                f"strings, got in index {idx}: {type(set_)}"
            )

        for word_idx, word in enumerate(set_):
            if not isinstance(word, str):
                raise TypeError(
                    "All set elements in a set of words should be strings. "
                    f"Got in set {idx} at position {word_idx}: {type(word)}"
                )

    if sets_name is not None and not isinstance(sets_name, str):
        raise TypeError(f"sets_name should be a string or None, got: {type(sets_name)}")
    if not isinstance(warn_lost_sets, bool):
        raise TypeError(f"warn_lost_sets should be a bool, got: {type(warn_lost_sets)}")
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose should be a bool, got: {type(verbose)}")

    embedding_sets: List[EmbeddingDict] = []

    # For each definitional pair:
    for set_idx, set_ in enumerate(sets):

        # Transform the pair to a embedding dict.
        # idea: (word_1, word_2) -> {'word_1': embedding, 'word_2'.: embedding}
        # TODO: Add identifier of the set that is being transformed.
        # if verbose:
        #     print(f"Transforming '{}' set ")
        not_found_words, embedding_pair = get_embeddings_from_set(
            model, set_, preprocessors, strategy, normalize, verbose
        )

        # If some word of the current pair can not be converted, discard the pair.
        if discard_incomplete_sets and len(not_found_words) > 0 and warn_lost_sets:
            set_name = f" of {sets_name} pair" if sets_name else ""
            logging.warning(
                f"Word(s) {not_found_words}{set_name} at index {set_idx} "
                "were not found. This pair will be omitted."
            )
        else:
            if normalize:

                for word in embedding_pair:
                    embedding = embedding_pair[word]
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    if np.linalg.norm(embedding) < 1:
                        normalized_embedding = embedding / np.linalg.norm(embedding)
                    embedding_pair[word] = normalized_embedding

            embedding_sets.append(embedding_pair)

    if len(embedding_sets) == 0:
        set_name = f"from the set {sets_name} " if sets_name else ""
        msg = (
            f"No set {set_name}could be converted to embedding because no set "
            "could be fully found in the model vocabulary."
        )
        raise Exception(msg)

    elif verbose:
        print(
            f"{len(embedding_sets)}/{len(sets)} sets of "
            "words were correctly converted to sets of embeddings"
        )

    return embedding_sets


def get_embeddings_from_query(
    model: WordEmbeddingModel,
    query: Query,
    lost_vocabulary_threshold: float = 0.2,
    preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
    strategy: str = "first",
    normalize: bool = False,
    warn_not_found_words: bool = False,
    verbose: bool = False,
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

    preprocessors : List[Dict[str, Union[str, bool, Callable]]]
        A list with preprocessor options.

        A ``preprocessor`` is a dictionary that specifies what processing(s) are
        performed on each word before it is looked up in the model vocabulary.
        For example, the ``preprocessor``
        ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
        and remove the accent from each word before searching for them in the
        model vocabulary. Note that an empty dictionary ``{}`` indicates that no
        preprocessing is done.

        The possible options for a preprocessor are:

        *   ``lowercase``: ``bool``. Indicates that the words are transformed to
            lowercase.
        *   ``uppercase``: ``bool``. Indicates that the words are transformed to
            uppercase.
        *   ``titlecase``: ``bool``. Indicates that the words are transformed to
            titlecase.
        *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
            the accents of the words are eliminated. The stripping type can be
            specified. True uses ‘unicode’ by default.
        *   ``preprocessor``: ``Callable``. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        ``[{}, {"lowercase": True, "strip_accents": True}]``
        ``{}`` allows first to search for the original words in the vocabulary of
        the model. In case some of them are not found,
        ``{"lowercase": True, "strip_accents": True}`` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : str, optional
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. all' will include all
        transformed words found, by default "first".

    normalize : bool, optional
        True indicates that embeddings will be normalized, by default False

    warn_not_found_words : bool, optional
        A flag that indicates if the function will warn (in the logger)
        the words that were not found in the model's vocabulary,
        by default False.

    verbose : bool, optional
        Indicates whether the execution status of this function is printed,
        by default False

    Returns
    -------
    Union[Tuple[EmbeddingSets, EmbeddingSets], None]
        A tuple of dictionaries containing the targets and attribute sets or None
        in case there is a set that has proportionally less embeddings than it was
        allowed to lose.
    """
    # Type check
    if not isinstance(query, Query):
        raise TypeError("query should be an instance of Query, got {}.".format(query))

    some_set_lost_more_words_than_threshold: bool = False

    target_embeddings: EmbeddingSets = {}
    attribute_embeddings: EmbeddingSets = {}

    # --------------------------------------------------------------------
    # get target sets embeddings
    for target_set, target_set_name in zip(query.target_sets, query.target_sets_names):
        not_found_words, obtained_embeddings = get_embeddings_from_set(
            model=model,
            word_set=target_set,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            verbose=verbose,
        )

        # warn not found words if it is enabled.
        _warn_not_found_words(
            warn_not_found_words, not_found_words, model.name, target_set_name
        )

        # if the lost words are greater than the threshold,
        # warn and change the flag.
        if _check_lost_vocabulary_threshold(
            model,
            obtained_embeddings,
            target_set,
            target_set_name,
            lost_vocabulary_threshold,
        ):
            some_set_lost_more_words_than_threshold = True

        target_embeddings[target_set_name] = obtained_embeddings

    # --------------------------------------------------------------------
    # get attribute sets embeddings
    for attribute_set, attribute_set_name in zip(
        query.attribute_sets, query.attribute_sets_names
    ):

        not_found_words, obtained_embeddings = get_embeddings_from_set(
            model=model,
            word_set=attribute_set,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            verbose=verbose,
        )

        _warn_not_found_words(
            warn_not_found_words, not_found_words, model.name, attribute_set_name
        )

        # if the filtered words are greater than the threshold,
        # log and change the flag.
        if _check_lost_vocabulary_threshold(
            model,
            obtained_embeddings,
            attribute_set,
            attribute_set_name,
            lost_vocabulary_threshold,
        ):
            some_set_lost_more_words_than_threshold = True

        attribute_embeddings[attribute_set_name] = obtained_embeddings

    # check if some set has fewer words than the threshold. if that's
    #  the case, return None
    if some_set_lost_more_words_than_threshold:
        logging.error(
            "At least one set of '{}' query has proportionally fewer embeddings "
            "than allowed by the lost_vocabulary_threshold parameter ({}). "
            "This query will return np.nan.".format(
                query.query_name, lost_vocabulary_threshold
            )
        )
        return None

    return target_embeddings, attribute_embeddings
