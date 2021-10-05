"""Embedding Coherence Test metric implementation."""
from typing import Any, Callable, Dict, List, Union

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class ECT(BaseMetric):
    """Embedding Coherence Test [1].

    The metric was originally proposed in [1] and implemented in [2].
    Values closer to 1 are better as they represent less bias.
    
    The general steps of the test, as defined in [1], are as follows:

    1. Embed all given target and attribute words with the given embedding model.
    2. Calculate mean vectors for the two sets of target word vectors.
    3. Measure the cosine similarity of the mean target vectors to all of the given
       attribute words.
    4. Calculate the Spearman r correlation between the resulting two lists of
       similarities.
    5. Return the correlation value as score of the metric (in the range of -1 to 1);
       higher is better.


    References
    ----------
    | [1]: Dev, S., & Phillips, J. (2019, April). Attenuating Bias in Word vectors.
    | [2]: https://github.com/sunipa/Attenuating-Bias-in-Word-Vec

    """

    # The metrics accepts two target sets and a single attribute set
    metric_template = (2, 1)
    metric_name = "Embedding Coherence Test"
    metric_short_name = "ECT"

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run ECT with the given query with the given parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to be
            tested.

        model : WordEmbeddingModel
            A word embedding model.

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

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. all' will include all
            transformed words found, by default "first".

        normalize : bool, optional
            True indicates that embeddings will be normalized, by default False

        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary, by default False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name and the result of the query.

        Examples
        --------
        >>> from wefe.metrics import ECT
        >>> from wefe.query import Query
        >>> from wefe.utils import load_test_model
        >>>
        >>> # define the query
        >>> query = Query(
        ...     target_sets=[
        ...         ["female", "woman", "girl", "sister", "she", "her", "hers",
        ...          "daughter"],
        ...         ["male", "man", "boy", "brother", "he", "him", "his", "son"],
        ...     ],
        ...     attribute_sets=[
        ...         [
        ...             "home", "parents", "children", "family", "cousins", "marriage",
        ...             "wedding", "relatives",
        ...         ],
        ...     ],
        ...     target_sets_names=["Female terms", "Male Terms"],
        ...     attribute_sets_names=["Family"],
        ... )
        >>> # load the model (in this case, the test model included in wefe)
        >>> model = load_test_model()
        >>>
        >>> # instance the metric and run the query
        >>> ECT().run_query(query, model) # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family',
        'result': 0.6190476190476191,
        'ect': 0.6190476190476191}
        >>> # if you want the embeddings to be normalized before calculating the metrics
        >>> # use the normalize parameter as True before executing the query.
        >>> ECT().run_query(query, model, normalize=True) # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family',
        'result': 0.7619047619047621,
        'ect': 0.7619047619047621}
        """
        # check the types of the provided arguments (only the defaults).
        self._check_input(query, model, locals())

        # transform query word sets into embeddings
        embeddings = get_embeddings_from_query(
            model=model,
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            warn_not_found_words=warn_not_found_words,
        )

        # If the lost vocabulary threshold is exceeded, return the default value
        if embeddings is None:
            return {"query_name": query.query_name, "result": np.nan, "ect": np.nan}

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        ect = self.__calculate_embedding_coherence(
            list(target_embeddings[0].values()),
            list(target_embeddings[1].values()),
            list(attribute_embeddings[0].values()),
        )

        return {"query_name": query.query_name, "result": ect, "ect": ect}

    def __calculate_embedding_coherence(
        self, target_set_1: list, target_set_2: list, attribute_set: list
    ) -> float:
        """Calculate the ECT metric over the given parameters. Return the result.

        Parameters
        ----------
        target_set_1 : list
            The first set of target words.
        target_set_2 : list
            The second set of target words.
        attribute_set : list
            The set of attribute words.

        Returns
        -------
        float
            The value denoting the Spearman correlation.
        """
        # Calculate mean vectors for both target vector sets
        target_means = [np.mean(s, axis=0) for s in (target_set_1, target_set_2)]

        # Measure similarities between mean vecotrs and all attribute words
        similarities = []
        for mean_vector in target_means:
            similarities.append([1 - cosine(mean_vector, a) for a in attribute_set])

        # Calculate similarity correlations
        return spearmanr(similarities[0], similarities[1]).correlation
