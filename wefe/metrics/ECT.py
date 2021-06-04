from typing import Any, Callable, Dict, List, Union
from wefe.preprocessing import get_embeddings_from_query
import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class ECT(BaseMetric):
    """An implementation of the Embedding Coherence Test.

    The metrics was originally proposed in [1] and implemented in [2].

    The general steps of the test, as defined in [1], are as follows:

    1. Embedd all given target and attribute words with the given embedding model
    2. Calculate mean vectors for the two sets of target word vectors
    3. Measure the cosine similarity of the mean target vectors to all of the given
    attribute words
    4. Calculate the Spearman r correlation between the resulting two lists of
    similarities
    5. Return the correlation value as score of the metric (in the range of -1 to 1);
    higher is better

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
        word_embedding: WordEmbeddingModel,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Runs ECT with the given query with the given parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to be
            tested.

        word_embedding_model : WordEmbeddingModel
            An object containing a word embeddings model.

        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is
            allowed to lose when transforming its words into embeddings.
            In the case that any set of the query loses proportionally more words
            than this limit, the result values will be np.nan, by default 0.2
        
        preprocessors : List[Dict[str, Union[str, bool, Callable]]]
            A list with preprocessor options.

            A dictionary of preprocessing options is a dictionary that specifies what
            transformations will be made to each word prior to being searched in the
            embeddings model. For example, `{'lowecase': True, 'strip_accents': True}` 
            will allow you to search for words in the word_set transformed to lowercase 
            and without accents.
            Note that an empty dictionary `{}` indicates that no transformation
            will be made to any word.

            A list of these preprocessor options will allow you to search for several
            variants of the words (depending on the search strategy) into the model.
            For example `[{}, {'lowecase': True, 'strip_accents': True}]` will allow you
            to search for each word first without any transformation and then transformed
            to lowercase and without accents.

            The available word preprocessing options are as follows (it is not necessary
            to put them all):

            - `lowercase`: `bool`. Indicates if the words are transformed to lowercase.
            - `uppercase`: `bool`. Indicates if the words are transformed to uppercase.
            - `titlecase`: `bool`. Indicates if the words are transformed to titlecase.
            - `strip_accents`: `bool`, `{'ascii', 'unicode'}`: Specifies if the accents of
                                the words are eliminated. The stripping type can be
                                specified. True uses 'unicode' by default.
            - `preprocessor`: `Callable`. It receives a function that operates on each
                            word. In the case of specifying a function, it overrides
                            the default preprocessor (i.e., the previous options
                            stop working).

            by default [{}]

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. all' will include all
            transformed words found., by default "first"
            
        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary
            , by default False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name and the result of the query.
        """

        # check the types of the provided arguments (only the defaults).
        self._check_input(query, word_embedding)

        # transform query word sets into embeddings
        embeddings = get_embeddings_from_query(
            model=word_embedding,
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            warn_not_found_words=warn_not_found_words,
        )

        # If the lost vocabulary threshold is exceeded, return the default value
        if embeddings is None:
            return {"query_name": query.query_name, "result": np.nan}

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
