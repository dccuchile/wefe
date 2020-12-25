import numpy as np
from typing import Any, Dict, Union

from ..query import Query
from ..word_embedding_model import PreprocessorArgs, WordEmbeddingModel
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

    metric_template = (2, 1)
    metric_name = 'Relative Norm Distance'
    metric_short_name = 'RND'

    def __calc_distance(self, vec1: np.ndarray, vec2: np.ndarray, distance_type='norm'):
        if distance_type == 'norm':
            return np.linalg.norm(np.subtract(vec1, vec2))
        elif distance_type == 'cos':
            c = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            return abs(c)
        else:
            raise Exception('Parameter distance_type can be either "norm" or "cos". '
                            'Given: {} '.format(distance_type))

    def __calc_rnd(self, target_0: np.ndarray, target_1: np.ndarray,
                   attribute: np.ndarray, attribute_words: list, distance_type: str,
                   average_distances: bool) -> Union[np.float64, list]:

        # calculates the average wv for the group words.
        target_1_avg_vector = np.average(target_0, axis=0)
        target_2_avg_vector = np.average(target_1, axis=0)

        sum_of_distances = 0
        distance_by_words = {}

        for attribute_word_index, attribute_embedding in enumerate(attribute):

            # calculate the distance
            current_distance = self.__calc_distance(
                attribute_embedding, target_1_avg_vector, distance_type=distance_type
            ) - self.__calc_distance(
                attribute_embedding, target_2_avg_vector, distance_type=distance_type)

            # add the distance of the neutral word to the accumulated
            # distances.
            sum_of_distances += current_distance
            # add the distance of the neutral word to the list of distances
            # by word
            distance_by_words[attribute_words[attribute_word_index]] = current_distance

        sorted_distances_by_word = {
            k: v
            for k, v in sorted(distance_by_words.items(), key=lambda item: item[1])
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
                  preprocessor_args: PreprocessorArgs = {
                      'strip_accents': False,
                      'lowercase': False,
                      'preprocessor': None,
                  },
                  secondary_preprocessor_args: PreprocessorArgs = None,
                  warn_not_found_words: bool = False,
                  *args: Any,
                  **kwargs: Any) -> Dict[str, Any]:
        """Calculate the RND metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets
            for be tested.

        word_embedding : 
            A  object that contain certain word embedding
            pretrained model.

        distance_type : str, optional
            Specifies which type of distance will be calculated. It could be:
            {norm, cos} , by default 'norm'.

        average_distances : bool, optional
            Specifies wheter the function averages the distances at the end of
            the calculations, by default True

        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is 
            allowed to lose when transforming its words into embeddings. 
            In the case that any set of the query loses proportionally more words 
            than this limit, the result values will be np.nan, by default 0.2

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
            , by default { 'strip_accents': False, 'lowercase': False, 'preprocessor': None, }
        
        secondary_preprocessor_args : PreprocessorArgs, optional
            Dictionary with the arguments that specify how the secondary pre-processing 
            of the words will be done, by default None.
            Indicates that in case a word is not found in the model's vocabulary 
            (using the default preprocessor or specified in preprocessor_args), 
            the function performs a second search for that word using the preprocessor 
            specified in this parameter.

        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary
            , by default False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name, the resulting score of the metric, 
            and a dictionary with the distances of each attribute word
            with respect to the target sets means.
        """
        # checks the types of the provided arguments (only the defaults).
        super().run_query(query, word_embedding, lost_vocabulary_threshold,
                          preprocessor_args, secondary_preprocessor_args,
                          warn_not_found_words, *args, **kwargs)

        # transforming query words into embeddings
        embeddings = word_embedding.get_embeddings_from_query(
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessor_args=preprocessor_args,
            secondary_preprocessor_args=secondary_preprocessor_args,
            warn_not_found_words=warn_not_found_words)

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {
                'query_name': query.query_name,
                'result': np.nan,
                "rnd": np.nan,
                "distances_by_word": {}
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        target_0_embeddings = list(target_embeddings[0].values())
        target_1_embeddings = list(target_embeddings[1].values())
        attribute_0_embeddings = list(attribute_embeddings[0].values())
        attribute_0_words = list(attribute_embeddings[0].keys())

        distance, distances_by_word = self.__calc_rnd(target_0_embeddings,
                                                      target_1_embeddings,
                                                      attribute_0_embeddings,
                                                      attribute_0_words, distance_type,
                                                      average_distances)

        return {
            "query_name": query.query_name,
            "result": distance,
            "rnd": distance,
            "distances_by_word": distances_by_word
        }
