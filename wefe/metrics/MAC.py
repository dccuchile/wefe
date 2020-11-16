import numpy as np
from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from .base_metric import BaseMetric


class MAC(BaseMetric):
    """A implementation of Mean Average Cosine Similarity (MAC).

    References
    -------
        [1] Thomas Manzini, Lim Yao Chong,Alan W Black, and Yulia Tsvetkov.
        Black is to criminalas caucasian is to police: Detecting and removing
        multiclass bias in word embeddings.
        In Proceedings of the 2019 Conference of the North American Chapter
        of the Association for Computational Linguistics:
        Human Lan-guage Technologies, Volume 1 (Long and Short Papers),
        pages 615–621,
        Minneapolis, Minnesota, June 2019. Association for
        Computational Linguistics.
    """
    def __init__(self):
        super().__init__((1, 'n'), 'Mean Average Cosine Similarity', 'MAC')

    def calc_s(self, t, A_j):
        def calc_cos_dist(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return 1 / len(A_j) * np.sum([calc_cos_dist(t, a) for a in A_j])

    def calc_mac(self, T, A):
        first_term = 1 / (len(T))
        mac = first_term * np.sum(
            [np.sum([self.calc_s(t_i, A_j) for A_j in A]) for t_i in T])

        return mac

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  lost_vocabulary_threshold: float = 0.2,
                  warn_filtered_words: bool = False):
        """Calculates the MAC metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute words for
            be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding
            pretrained model.
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount
            of words in any word set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered
            words, by default False.
        """

        # Standard input procedure: check the inputs and obtain the
        # embeddings.
        embeddings = super().run_query(query, word_embedding, lost_vocabulary_threshold)

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # get the target and attribute embeddings dicts
        target_embeddings_dict, attribute_embeddings_dict = embeddings
        target_0_embeddings = list(target_embeddings_dict[0].values())
        attribute_embeddings_all_sets = [
            list(target_dict.values()) for target_dict in attribute_embeddings_dict
        ]

        result = self.calc_mac(target_0_embeddings, attribute_embeddings_all_sets)

        return {'query_name': query.query_name_, 'result': result}
