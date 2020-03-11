import numpy as np
import logging

from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from ..utils import get_embeddings_from_word_set, verify_metric_input, verify_vocabulary_threshold


class RND:
    """A implementation of Relative Norm Distance. 
    It measures the relative strength of association of a set of neutral words with respect to two groups.

    References
    -------
    Nikhil Garg, Londa Schiebinger, Dan Ju-rafsky, and James Zou.   
    Word embeddings quantify 100 years of gender and ethnic stereotypes.
    Proceedings of the National Academy of Sciences, 115(16):E3635–E3644,2018.
    """

    def __init__(self):
        self.template_required = (2, 1)
        self.method_name = 'Relative Norm Distance (RND)'
        self.abbreviated_method_name = 'RND'

    def __calc_distance_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray, distance_type='norm'):
        """Calculates the cos distance between two embedding vectors
        
        Parameters
        ----------
        vec1 : np.ndarray
            First embedding vector.
        vec2 : np.ndarray
            Second embedding vector.
        distance_type : str, optional
            Indicates which type of distance will be calculated. It could be: {norm, cos} , by default 'norm'
        
        Returns
        -------
        np.float
            The calculated distance between the given vectors.
        
        Raises
        ------
        Exception
            if distance_type is not norm or cos.
        """

        if distance_type == 'norm':
            return np.linalg.norm(np.subtract(vec1, vec2))
        elif distance_type == 'cos':
            c = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            return abs(c)
        else:
            raise Exception('Parameter distance_type can be either "norm" or "cos". Given: {} '.format(distance_type))

    def __calc_relative_norm_distance(self, word_embedding: WordEmbeddingModel, attribute_words: list,
                                      target_1_embeddings, target_2_embeddings, attribute_embeddings,
                                      distance_type: str, average_distances: bool, disable_vocab_warnings: bool):
        """Captures the relative strength of association of a set of neutral words with respect to two groups.
        
        Parameters
        ----------
        word_embedding : WordEmbeddingModel
            The word embedding model used
        attribute_words : [type]
            The filtered attribute word set.
        attribute_lost_words : [type]
            The attribute words that were filtered.
        target_1_embeddings : [type]
            The embeddings that represent the target 1 words.
        target_2_embeddings : [type]
            The embeddings that represent the target 2 words.
        attribute_embeddings : [type]
            The embeddings that represent the target 2 words.
        distance_type : str
            Indicates which type of distance will be calculated.
        average_distances : bool
            Indicates if the function averages the distances at the end of the calculations.
        disable_vocab_warnings : bool
            A flag that indicates if the function will warn about the filtered words, by default False.
        """

        # calculates the average wv for the group words.
        target_1_avg_vector = np.average(target_1_embeddings, axis=0)
        target_2_avg_vector = np.average(target_2_embeddings, axis=0)

        sum_of_distances = 0
        distance_by_words = []

        for attribute_word_index, attribute_embedding in enumerate(attribute_embeddings):

            # calculate the distance
            current_distance = self.__calc_distance_between_vectors(
                attribute_embedding, target_1_avg_vector,
                distance_type=distance_type) - self.__calc_distance_between_vectors(
                    attribute_embedding, target_2_avg_vector, distance_type=distance_type)

            # add the distance of the neutral word to the accumulated distances.
            sum_of_distances += current_distance
            # add the distance of the neutral word to the list of distances by word
            distance_by_words.append([attribute_words[attribute_word_index], current_distance])

        # calculate the average of the distances and return
        if average_distances == True:
            return sum_of_distances / len(distance_by_words), sorted(distance_by_words, key=lambda x: x[1])

        return sum_of_distances, sorted(distance_by_words, key=lambda x: x[1])

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  lost_vocabulary_threshold: float = 0.2,
                  distance_type: str = 'norm',
                  average_distances: bool = True,
                  disable_vocab_warnings: bool = True):
        """Calculates the RND metric over the provided parameters. 
        
        Parameters
        ----------
        query : Query
            The query to be tested
        word_embedding : WordEmbeddingModel
            The embedding model from which the vectors will be extracted
        lost_vocabulary_threshold : float, optional
            Indicates when a test is invalid due the loss of certain amount of words in any word set., by default 0.2
        distance_type : str, optional
            Indicates which type of distance will be calculated. It could be: {norm, cos} , by default 'norm'
        average_distances : bool, optional
            Indicates if the function averages the distances at the end of the calculations. by default, true.
        disable_vocab_warnings : bool, optional
            [description], by default True
        """

        # check the inputs
        verify_metric_input(query, word_embedding, self.template_required, self.method_name)

        # extracts the sets and the names
        target_sets = query.target_sets
        attribute_sets = query.attribute_sets
        target_sets_names = query.target_sets_names
        attribute_sets_names = query.attribute_sets_names

        word_embedding_name = word_embedding.model_name

        # filter the word vectors, leaving only the words existing words in the wv vocab.
        target_1_embeddings, target_1_lost_words = get_embeddings_from_word_set(
            target_sets[0], word_embedding, warn_filtered_words=disable_vocab_warnings)
        target_2_embeddings, target_2_lost_words = get_embeddings_from_word_set(
            target_sets[1], word_embedding, warn_filtered_words=disable_vocab_warnings)

        attribute_embeddings, attribute_lost_words = get_embeddings_from_word_set(
            attribute_sets[0], word_embedding, warn_filtered_words=disable_vocab_warnings)

        # verify the missing vocabulary thresholds
        if not verify_vocabulary_threshold(target_sets[0], target_1_lost_words, lost_vocabulary_threshold,
                                           target_sets_names[0], word_embedding_name):
            return np.nan, []

        if not verify_vocabulary_threshold(target_sets[0], target_2_lost_words, lost_vocabulary_threshold,
                                           target_sets_names[1], word_embedding_name):
            return np.nan, []

        if not verify_vocabulary_threshold(attribute_sets[0], attribute_lost_words, lost_vocabulary_threshold,
                                           attribute_sets_names[0], word_embedding_name):
            return np.nan, []

        # calculate the relative norm distance
        filtered_attribute_words = list(filter(lambda x: x not in attribute_lost_words, attribute_sets[0]))
        avg_distance, distances_by_word = self.__calc_relative_norm_distance(word_embedding, filtered_attribute_words,
                                                                             target_1_embeddings, target_2_embeddings,
                                                                             attribute_embeddings, distance_type,
                                                                             average_distances, disable_vocab_warnings)

        return {
            "exp_name": "{} vs {} wrt {}".format(target_sets_names[0], target_sets_names[1], attribute_sets_names[0]),
            "result": avg_distance,
            "results_by_word": distances_by_word
        }
