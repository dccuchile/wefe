import numpy as np

from .base_metric import BaseMetric
from ..word_embedding_model import WordEmbeddingModel
from ..query import Query


class WEAT(BaseMetric):
    """A implementation of Word Embedding Association Test (WEAT). 
    
    It measures the degree of association between two sets of target words and two sets of attribute words through a permutation test.  
    
    References
    ----------
    Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. 
    Semantics derived automatically from language corpora contain human-like biases.
    Science,356(6334):183–186, 2017.
    """
    def __init__(self):
        super().__init__((2, 2), 'Word Embedding Association Test', 'WEAT')

    def __calc_s(self, w, A, B):
        def calc_cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        A_cosine_distance = np.array([calc_cos_sim(w, a) for a in A])
        B_cosine_distance = np.array([calc_cos_sim(w, b) for b in B])
        return np.mean(A_cosine_distance) - np.mean(B_cosine_distance)

    def __calc_effect_size(self, X, Y, A, B):
        first_term = np.mean(np.array([self.__calc_s(x, A, B) for x in X]))
        second_term = np.mean(np.array([self.__calc_s(y, A, B) for y in Y]))

        std_dev = np.std(
            np.array([self.__calc_s(w, A, B) for w in np.concatenate((X, Y))]))

        return (first_term - second_term) / std_dev

    def __calc_weat(self, X, Y, A, B):
        first_term = np.array([self.__calc_s(x, A, B) for x in X])
        second_term = np.array([self.__calc_s(y, A, B) for y in Y])
        return np.sum(first_term) - np.sum(second_term)

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  return_effect_size: bool = False,
                  lost_vocabulary_threshold: bool = 0.2,
                  warn_filtered_words: bool = False) -> dict:
        """Calculates the WEAT metric over the provided parameters. 
        
        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute words for be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding pretrained model.
        return_effect_size : bool, optional
            Indicates if the returned result is the effect size, by default False.
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount of words in any word set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered words, by default False.

        Returns
        -------
        dict
            A dictionary with the query name and the result of the query.
        """

        # Standard input procedure: check the entries and obtain the embeddings.
        embeddings = self._get_embeddings_from_query(
            query, word_embedding, warn_filtered_words,
            lost_vocabulary_threshold)

        # if there is any/some set has less words than the allowed limit, return the default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # get the target and attribute embeddings
        target_embeddings, attribute_embeddings = embeddings

        target_0 = list(target_embeddings[0].values())
        target_1 = list(target_embeddings[1].values())
        attribute_0 = list(attribute_embeddings[0].values())
        attribute_1 = list(attribute_embeddings[1].values())

        # if the requested value is the effect size:
        if return_effect_size:
            result = self.__calc_effect_size(target_0, target_1, attribute_0,
                                             attribute_1)
            return {'query_name': query.query_name_, 'result': result}

        result = self.__calc_weat(target_0, target_1, attribute_0, attribute_1)
        return {'query_name': query.query_name_, 'result': result}
