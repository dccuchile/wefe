import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from .base_metric import BaseMetric
from ..query import Query
from ..word_embedding_model import WordEmbeddingModel


class ECT(BaseMetric):
    """An implementation of the Embedding Coherence Test.

    The metrics was originally proposed in [1] and implemented in [2].

    The general steps of the test, as defined in [1], are as follows:

    1. Embedd all given target and attribute words with the given embedding model
    2. Calculate mean vectors for the two sets of target word vectors
    3. Measure the cosine similarity of the mean target vectors to all of the given attribute words
    4. Calculate the Spearman r correlation between the resulting two lists of similarities
    5. Return the correlation value as score of the metric (in the range of -1 to 1); higher is
       better


    References
    ----------
    | [1]: Dev, S., & Phillips, J. (2019, April). Attenuating Bias in Word vectors.
    | [2]: https://github.com/sunipa/Attenuating-Bias-in-Word-Vec
    """
    def __init__(self):
        """
        Initialize the metric.

        Args:
            self: (todo): write your description
        """
        # The metrics accepts two target sets and a single attribute set
        metric_template = (2, 1)
        metric_name = "Embedding Coherence Test"
        metric_short_name = "ECT"

        super().__init__(metric_template, metric_name, metric_short_name)

    def run_query(
            self,
            query: Query,
            word_embedding: WordEmbeddingModel,
            lost_vocabulary_threshold: float = 0.2,
            warn_filtered_words: bool = True):
        """Runs the given query with the given parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute words for be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding pretrained model.
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount of words in any word
            set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered words, by default
            False.

        Returns
        -------
        dict
            A dictionary with the query name and the result of the query.
        """

        # Get word vectors from the specified query
        embeddings = self._get_embeddings_from_query(
            query,
            word_embedding,
            warn_filtered_words=warn_filtered_words,
            lost_vocabulary_threshold=lost_vocabulary_threshold)

        # If the lost vocabulary threshold is exceeded, return the default value
        if embeddings is None:
            return {"query_name": query.query_name_, "result": np.nan}

        return {
            "query_name": query.query_name_,
            "result": self.__calculate_embedding_coherence(
                list(embeddings[0][0].values()),
                list(embeddings[0][1].values()),
                list(embeddings[1][0].values()))}

    def __calculate_embedding_coherence(
            self,
            target_set_1: list,
            target_set_2: list,
            attribute_set: list) -> float:
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
