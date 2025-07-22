"""Module with utilities that ease the transformation of word sets to embeddings."""

import logging
from collections.abc import Sequence
from typing import Callable, Literal, Optional, Union

import numpy as np
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

# --- Type Aliases ---
EmbeddingDict = dict[str, np.ndarray]
EmbeddingSets = dict[str, EmbeddingDict]


# --- Constants for Preprocessing Options ---
class PreprocessingKeys:
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    TITLECASE = "titlecase"
    STRIP_ACCENTS = "strip_accents"
    PREPROCESSOR = "preprocessor"


def _apply_case_transformation(
    word: str,
    options: dict[str, Union[str, bool, Callable[[str], str]]],
) -> str:
    """
    Applies a case transformation to a given word based on provided options.

    Parameters
    ----------
    word : str
        The input word to be transformed.
    options : dict[str, Union[str, bool, Callable[[str], str]]]
        A dictionary containing case transformation options. Supported keys are:
        - PreprocessingKeys.LOWERCASE: If True, convert word to lowercase.
        - PreprocessingKeys.UPPERCASE: If True, convert word to uppercase.
        - PreprocessingKeys.TITLECASE: If True, convert word to title case.

    Returns
    -------
    str
        The transformed word according to the specified case option.
        If no case option is set, returns the original word.
    """

    if options.get(PreprocessingKeys.LOWERCASE):
        return word.lower()
    if options.get(PreprocessingKeys.UPPERCASE):
        return word.upper()
    if options.get(PreprocessingKeys.TITLECASE):
        return word.title()
    return word


def _apply_accent_stripping(word: str, strip_accents_option: Union[bool, str]) -> str:
    """Aplica la eliminación de acentos a una palabra."""
    if strip_accents_option is True or strip_accents_option == "unicode":
        return strip_accents_unicode(word)
    if strip_accents_option == "ascii":
        return strip_accents_ascii(word)
    return word


def preprocess_word(
    word: str,
    options: dict[str, Union[str, bool, Callable]] = {},
    vocab_prefix: Optional[str] = None,
) -> str:
    """pre-processes a word before it is searched in the model's vocabulary.

    Parameters
    ----------
    word : str
        Word to be preprocessed.
    options : Dict[str, Union[str, bool, Callable]]
        Dictionary with arguments that specifies how the words will be preprocessed,
        The available word preprocessing options are as follows:

        - `lowercase`: bool. Indicates if the words are transformed to lowercase.
        - `uppercase`: bool. Indicates if the words are transformed to uppercase.
        - `titlecase`: bool. Indicates if the words are transformed to titlecase.
        - `strip_accents`: `bool`, `{'ascii', 'unicode'}`: Specifies if the
          accents of the words are eliminated. The stripping type can be
          specified. True uses 'unicode' by default.
        - `preprocessor`: Callable. It receives a function that operates on each
          word. In the case of specifying a function, it overrides
          the default preprocessor (i.e., the previous options
          stop working).

        By default, no preprocessing is generated, which is equivalent to `{}`

    Returns
    -------
    str
        The pre-processed word according to the given parameters.

    """
    preprocessor_func = options.get(PreprocessingKeys.PREPROCESSOR)

    # If a custom preprocessor is specified, it takes precedence.
    if callable(preprocessor_func):
        word = preprocessor_func(word)
    else:
        word = _apply_case_transformation(word=word, options=options)
        word = _apply_accent_stripping(
            word=word,
            strip_accents_option=options.get(PreprocessingKeys.STRIP_ACCENTS, False),
        )

    if vocab_prefix:
        return vocab_prefix + word
    return word


def get_embeddings_from_set(
    model: WordEmbeddingModel,
    word_set: Sequence[str],
    preprocessors: list[dict[str, str | bool | Callable]],
    strategy: Literal["first", "all"] = "first",
    normalize: bool = False,
    verbose: bool = False,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Transform a sequence of words into dictionary that maps word - word embedding.

    The method discards out words that are not in the model's vocabulary
    (according to the rules specified in the preprocessors).


    Parameters
    ----------
    model : WordEmbeddingModel
        A word embedding model

    word_set : Sequence[str]
        A sequence with the words that this function will convert to embeddings.

    preprocessors : list[dict[str, str | bool | Callable]]
        A list with preprocessor options.

        A `preprocessor` is a dictionary that specifies what processing(s) are
        performed on each word before it is looked up in the model vocabulary.
        For example, the `preprocessor`
        `{'lowecase': True, 'strip_accents': True}` allows you to lowercase
        and remove the accent from each word before searching for them in the
        model vocabulary. Note that an empty dictionary `{}` indicates that no
        preprocessing is done.

        The possible options for a preprocessor are:

        * `lowercase`: `bool`. Indicates that the words are transformed to
            lowercase.
        * `uppercase`: `bool`. Indicates that the words are transformed to
            uppercase.
        * `titlecase`: `bool`. Indicates that the words are transformed to
            titlecase.
        * `strip_accents`: `bool`, `{'ascii', 'unicode'}`: Specifies that
            the accents of the words are eliminated. The stripping type can be
            specified. True uses 'unicode' by default.
        * `preprocessor`: `Callable`. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        `[{}, {"lowercase": True, "strip_accents": True}]`
        `{}` allows searching first for the original words in the vocabulary of
        the model. In case some of them are not found,
        `{"lowercase": True, "strip_accents": True}` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : Literal["first", "all"]
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. 'all' will include all
        transformed words found, by default "first".

    normalize : bool, optional
        True indicates that embeddings will be normalized, by default False

    verbose : bool, optional
        Indicates whether the execution status of this function is printed,
        by default False

    Returns
    -------
    Tuple[List[str], EmbeddingDict]
        A tuple containing the words that could not be found and a dictionary with
        the found words and their corresponding embeddings.


    """
    # ----------------------------------------------------------------------------------
    # type verifications.

    if not isinstance(model, WordEmbeddingModel):
        raise TypeError(f"model must be a WordEmbeddingModel instance, got {model}.")
    if not isinstance(word_set, (list, tuple, np.ndarray)):
        raise TypeError(
            f"word_set must be a list, tuple or np.array of strings, got {word_set}."
        )
    if not isinstance(preprocessors, list):
        raise TypeError(
            "preprocessors must be a list of dicts which contains preprocessor "
            f"options, got {preprocessors}."
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
                f"each preprocessor must be a dict, got {p} at index {idx}."
            )
    if strategy != "first" and strategy != "all":
        raise ValueError(f"strategy must be 'first' or 'all', got {strategy}.")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a boolean, got {type(normalize)}.")
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be a boolean, got {type(verbose)}.")
    # ----------------------------------------------------------------------------------
    # filter the words

    selected_embeddings: EmbeddingDict = {}
    not_found_words: list[str] = []

    for original_word in word_set:
        found_current_word = False
        for preprocessor_options in preprocessors:
            processed_word = preprocess_word(
                original_word,
                options=preprocessor_options,
                vocab_prefix=model.vocab_prefix,
            )
            embedding = model.get(processed_word)

            if embedding is not None:
                selected_embeddings[processed_word] = embedding
                found_current_word = True
                if strategy == "first":
                    break  # Stop at the first found embedding

            if not found_current_word and processed_word not in not_found_words:
                not_found_words.append(processed_word)

    if normalize and selected_embeddings:
        keys = list(selected_embeddings.keys())
        values = np.array(list(selected_embeddings.values()))
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        normalized_values = np.where(norms == 0, values, values / norms)
        selected_embeddings = dict(zip(keys, normalized_values))

    if verbose:
        print(
            f"Word(s) found: {list(selected_embeddings.keys())}, "
            f"not found: {not_found_words}"
        )

    return not_found_words, selected_embeddings


def _warn_not_found_words(
    warn_not_found_words: bool,
    not_found_words: list[str],
    model_name: str,
    set_name: str,
) -> None:
    if not isinstance(warn_not_found_words, bool):
        raise TypeError(
            f"warn_not_found_words must be a boolean, got {warn_not_found_words}."
        )

    if warn_not_found_words and len(not_found_words) > 0:
        logging.warning(
            f"The following words from set '{set_name}' do not exist within the "
            f"vocabulary of {model_name}: {not_found_words}"
        )


def _check_lost_vocabulary_threshold(
    model: WordEmbeddingModel,
    embeddings: EmbeddingDict,
    word_set: Sequence[str],
    word_set_name: str,
    lost_vocabulary_threshold: float,
) -> bool:
    if not isinstance(lost_vocabulary_threshold, (float, np.floating)):
        raise TypeError(
            f"lost_vocabulary_threshold must be float, got {lost_vocabulary_threshold}."
        )
    if not (0.0 <= lost_vocabulary_threshold <= 1.0):
        raise ValueError("lost_vocabulary_threshold must be between 0.0 and 1.0.")

    total_words = len(word_set)
    if total_words == 0:
        logging.warning(
            f"The word set '{word_set_name}' is empty. Threshold check skipped."
        )
        return False  # No words lost if the set is empty

    number_of_found_words = len(embeddings)
    number_of_lost_words = total_words - number_of_found_words
    percentage_of_lost_words = number_of_lost_words / total_words

    # if the percentage of filtered words are greater than the
    # threshold, log and return False
    if percentage_of_lost_words > lost_vocabulary_threshold:
        logging.warning(
            f"The transformation of '{word_set_name}' into {model.name} embeddings "
            f"lost proportionally more words than specified in "
            f"'lost_vocabulary_threshold':{percentage_of_lost_words:.2%} lost with "
            f"respect to {lost_vocabulary_threshold:.2%} maximum loss allowed."
        )
        return True
    return False


def get_embeddings_from_tuples(
    model: WordEmbeddingModel,
    sets: Sequence[Sequence[str]],
    preprocessors: list[dict[str, str | bool | Callable]],
    sets_name: Union[str, None] = None,
    strategy: Literal["first", "all"] = "first",
    normalize: bool = False,
    discard_incomplete_sets: bool = True,
    warn_lost_sets: bool = True,
    verbose: bool = False,
) -> list[EmbeddingDict]:
    """Given a sequence of word sets, obtain their corresponding embeddings.

    Parameters
    ----------
    model : WordEmbeddingModel
        A word embedding model.

    sets : Sequence[Sequence[str]]
        A sequence containing word sets.
        Example: `[['woman', 'man'], ['she', 'he'], ['mother', 'father'] ...]`.

    sets_name : Union[str, optional]
        The name of the set of word sets. Example: `defining sets`.
        This parameter is used only for printing.
        by default None

    preprocessors : list[dict[str, str | bool | Callable]]
        A list with preprocessor options.

        A `preprocessor` is a dictionary that specifies what processing(s) are
        performed on each word before it is looked up in the model vocabulary.
        For example, the `preprocessor`
        `{'lowecase': True, 'strip_accents': True}` allows you to lowercase
        and remove the accent from each word before searching for them in the
        model vocabulary. Note that an empty dictionary `{}` indicates that no
        preprocessing is done.

        The possible options for a preprocessor are:

        * `lowercase`: `bool`. Indicates that the words are transformed to
            lowercase.
        * `uppercase`: `bool`. Indicates that the words are transformed to
            uppercase.
        * `titlecase`: `bool`. Indicates that the words are transformed to
            titlecase.
        * `strip_accents`: `bool`, `{'ascii', 'unicode'}`: Specifies that
            the accents of the words are eliminated. The stripping type can be
            specified. True uses 'unicode' by default.
        * `preprocessor`: `Callable`. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        `[{}, {"lowercase": True, "strip_accents": True}]`
        `{}` allows searching first for the original words in the vocabulary of
        the model. In case some of them are not found,
        `{"lowercase": True, "strip_accents": True}` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : Literal["first", "all"]
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. 'all' will include all
        transformed words found, by default "first".

    normalize : bool, optional
        True indicates that embeddings will be normalized, by default False

    discard_incomplete_sets : bool, optional
        True indicates that if a set could not be completely converted, it will be
        discarded. by default True

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
    # --- Type and Value Validations ---
    if not isinstance(model, WordEmbeddingModel):
        raise TypeError(
            f"model must be a WordEmbeddingModel instance, got {type(model)}."
        )

    if not isinstance(sets, (list, tuple, np.ndarray)):
        raise TypeError(
            "sets must be a sequence of sequences (list, tuple or np.array) "
            f"of strings, got: {type(sets)}."
        )

    for idx, set_ in enumerate(sets):
        if not isinstance(set_, (list, tuple, np.ndarray)):
            raise TypeError(
                "Every set in sets must be a list, tuple or np.array of "
                f"strings, got in index {idx}: {type(set_)}"
            )

        for word_idx, word in enumerate(set_):
            if not isinstance(word, str):
                raise TypeError(
                    "All set elements in a set of words must be strings. "
                    f"Got in set {idx} at position {word_idx}: {type(word)}"
                )

    if sets_name is not None and not isinstance(sets_name, str):
        raise TypeError(f"sets_name must be a string or None, got: {type(sets_name)}")
    if not isinstance(warn_lost_sets, bool):
        raise TypeError(f"warn_lost_sets must be a bool, got: {type(warn_lost_sets)}")
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be a bool, got: {type(verbose)}")

    embedding_sets: list[EmbeddingDict] = []

    # For each definitional pair:
    for set_idx, set_ in enumerate(sets):
        # Transform the pair to a embedding dict.
        # idea: (word_1, word_2) -> {'word_1': embedding, 'word_2'.: embedding}
        # TODO: Add identifier of the set that is being transformed.
        not_found_words, embedding_pair = get_embeddings_from_set(
            model=model,
            word_set=set_,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            verbose=verbose,
        )

        # If some word of the current pair can not be converted, discard the pair.
        if discard_incomplete_sets and not_found_words:
            set_name = f" of {sets_name} pair" if sets_name else ""
            logging.warning(
                f"Word(s) {not_found_words}{set_name} at index {set_idx} "
                "were not found. This pair will be omitted."
            )
        else:
            embedding_sets.append(embedding_pair)

    if not embedding_sets:
        set_name_for_msg = f" from the set '{sets_name}'" if sets_name else ""
        msg = (
            f"No set{set_name_for_msg} could be converted to embedding because no set "
            "could be fully found in the model vocabulary or all were discarded."
        )
        raise ValueError(msg)

    if verbose:
        print(
            f"{len(embedding_sets)}/{len(sets)} sets of "
            "words were correctly converted to sets of embeddings."
        )

    return embedding_sets


def get_embeddings_from_query(
    model: WordEmbeddingModel,
    query: Query,
    lost_vocabulary_threshold: float,
    preprocessors: list[dict[str, str | bool | Callable]],
    strategy: Literal["first", "all"] = "first",
    normalize: bool = False,
    warn_not_found_words: bool = False,
    verbose: bool = False,
) -> Union[tuple[EmbeddingSets, EmbeddingSets], None]:
    """Obtain the word vectors associated with the provided Query.

    The words that do not appear in the word embedding pretrained model
    vocabulary under the specified pre-processing are discarded.
    If the remaining words percentage in any query set is lower than the specified
    threshold, the function will return None.


    Parameters
    ----------
    query : Query
        The query to be processed.

    lost_vocabulary_threshold : float
        Indicates the proportional limit of words that any set of the query is
        allowed to lose when transforming its words into embeddings.
        In the case that any set of the query loses proportionally more words
        than this limit, this method will return None.

    preprocessors : list[dict[str, str | bool | Callable]]
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
            specified. True uses 'unicode' by default.
        *   ``preprocessor``: ``Callable``. It receives a function that operates
            on each word. In the case of specifying a function, it overrides the
            default preprocessor (i.e., the previous options stop working).

        A list of preprocessor options allows you to search for several
        variants of the words into the model. For example, the preprocessors
        ``[{}, {"lowercase": True, "strip_accents": True}]``
        ``{}`` allows searching first for the original words in the vocabulary of
        the model. In case some of them are not found,
        ``{"lowercase": True, "strip_accents": True}`` is executed on these words
        and then they are searched in the model vocabulary.
        by default [{}]

    strategy : str, optional
        The strategy indicates how it will use the preprocessed words: 'first' will
        include only the first transformed word found. 'all' will include all
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
        raise TypeError(f"query must be an instance of Query, got {query}.")
    if not preprocessors:
        raise ValueError(
            "preprocessors must contain at least one preprocessor dictionary "
            "(e.g., [{}])."
        )
    some_set_lost_more_words_than_threshold: bool = False

    target_embeddings: EmbeddingSets = {}
    attribute_embeddings: EmbeddingSets = {}

    # --------------------------------------------------------------------
    # get target sets embeddings
    for target_set, target_set_name in zip(query.target_sets, query.target_sets_names):
        not_found_words, obtained_embeddings = get_embeddings_from_set(
            model=model,
            word_set=target_set,
            preprocessors=[{}] if preprocessors is None else preprocessors,
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
            f"At least one set of '{query.query_name}' query has proportionally fewer "
            "embeddings than allowed by the lost_vocabulary_threshold parameter "
            f"({lost_vocabulary_threshold}). This query will return np.nan."
        )
        return None

    return target_embeddings, attribute_embeddings
