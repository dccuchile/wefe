import numpy as np
from typing import Union

from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from .base_metric import BaseMetric


class RND(BaseMetric):
    """A implementation of Relative Norm Distance (RND).

    It measures the relative strength of association of a set of neutral words
    with respect to two groups.

    References
    ----------
    Nikhil Garg, Londa Schiebinger, Dan Ju-rafsky, and James Zou.
    Word embeddings quantify 100 years of gender and ethnic stereotypes.
    Proceedings of the National Academy of Sciences, 115(16):E3635–E3644,2018.
    """
    def __init__(self):
        super().__init__((2, 1), 'Relative Norm Distance', 'RND')

    def __calc_distance(self,
                        vec1: np.ndarray,
                        vec2: np.ndarray,
                        distance_type='norm'):
        if distance_type == 'norm':
            return np.linalg.norm(np.subtract(vec1, vec2))
        elif distance_type == 'cos':
            c = np.dot(vec1,
                       vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            return abs(c)
        else:
            raise Exception(
                'Parameter distance_type can be either "norm" or "cos". '
                'Given: {} '.format(distance_type))

    def __calc_rnd(self, target_0: np.ndarray, target_1: np.ndarray,
                   attribute: np.ndarray, attribute_words: list,
                   distance_type: str, average_distances: bool,
                   disable_vocab_warnings: bool) -> Union[np.float64, list]:

        # calculates the average wv for the group words.
        target_1_avg_vector = np.average(target_0, axis=0)
        target_2_avg_vector = np.average(target_1, axis=0)

        sum_of_distances = 0
        distance_by_words = {}

        for attribute_word_index, attribute_embedding in enumerate(attribute):

            # calculate the distance
            current_distance = self.__calc_distance(
                attribute_embedding,
                target_1_avg_vector,
                distance_type=distance_type) - self.__calc_distance(
                    attribute_embedding,
                    target_2_avg_vector,
                    distance_type=distance_type)

            # add the distance of the neutral word to the accumulated
            # distances.
            sum_of_distances += current_distance
            # add the distance of the neutral word to the list of distances
            # by word
            distance_by_words[
                attribute_words[attribute_word_index]] = current_distance

        sorted_distances_by_word = {
            k: v
            for k, v in sorted(distance_by_words.items(),
                               key=lambda item: item[1])
        }

        if average_distances:
            # calculate the average of the distances and return
            mean_distance = sum_of_distances / len(distance_by_words)
            return mean_distance, sorted_distances_by_word

        return sum_of_distances, sorted_distances_by_word

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  distance_type: str = 'norm',
                  average_distances: bool = True,
                  lost_vocabulary_threshold: float = 0.2,
                  warn_filtered_words: bool = True) -> dict:
        """Calculates the RND metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute words
            for be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding
            pretrained model.
        distance_type : str, optional
            Indicates which type of distance will be calculated. It could be:
            {norm, cos} , by default 'norm'
        average_distances : bool, optional
            Indicates if the function averages the distances at the end of
            the calculations. by default, true.
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount
            of words in any word set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered
            words, by default False.

        Returns
        -------
        dict
            A dictionary with the query name, the result of the query
            and a dictionary with the distances of each attribute word
            with respect to the target sets means.
        """

        # Standard input procedure: check the inputs and obtain the
        # embeddings.
        embeddings = super().run_query(query, word_embedding,
                                       lost_vocabulary_threshold)

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # get the target and attribute embeddings
        target_embeddings_dict, attribute_embeddings_dict = embeddings
        target_0_embeddings = list(target_embeddings_dict[0].values())
        target_1_embeddings = list(target_embeddings_dict[1].values())
        attribute_embeddings = list(attribute_embeddings_dict[0].values())
        attribute_words = list(attribute_embeddings_dict[0].keys())

        distance, distances_by_word = self.__calc_rnd(
            target_0_embeddings, target_1_embeddings, attribute_embeddings,
            attribute_words, distance_type, average_distances,
            warn_filtered_words)

        return {
            "query_name": query.query_name_,
            "result": distance,
            "results_by_word": distances_by_word
        }
