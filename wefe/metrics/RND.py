import numpy as np
from typing import Any, Callable, Dict, List, Union

from wefe.utils import cosine_similarity
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query


class RND(BaseMetric):
    """A implementation of Relative Norm Distance (RND).

    It measures the relative strength of association of a set of neutral words
    with respect to two groups.

    References
    ----------
    Nikhil Garg, Londa Schiebinger, Dan Jurafsky, and James Zou.
    Word embeddings quantify 100 years of gender and ethnic stereotypes.
    Proceedings of the National Academy of Sciences, 115(16):E3635–E3644,2018.
    """

    metric_template = (2, 1)
    metric_name = "Relative Norm Distance"
    metric_short_name = "RND"

    def __calc_distance(self, vec1: np.ndarray, vec2: np.ndarray, distance_type="norm"):
        if distance_type == "norm":
            return np.linalg.norm(np.subtract(vec1, vec2))
        elif distance_type == "cos":
            # c = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            c = cosine_similarity(vec1, vec2)
            return abs(c)
        else:
            raise Exception(
                'Parameter distance_type can be either "norm" or "cos". '
                "Given: {} ".format(distance_type)
            )

    def __calc_rnd(
        self,
        target_0: np.ndarray,
        target_1: np.ndarray,
        attribute: np.ndarray,
        attribute_words: list,
        distance_type: str,
        average_distances: bool,
    ) -> Union[np.float64, list]:

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
                attribute_embedding, target_2_avg_vector, distance_type=distance_type
            )

            # add the distance of the neutral word to the accumulated
            # distances.
            sum_of_distances += current_distance
            # add the distance of the neutral word to the list of distances
            # by word
            distance_by_words[attribute_words[attribute_word_index]] = current_distance

        sorted_distances_by_word = {
            k: v for k, v in sorted(distance_by_words.items(), key=lambda item: item[1])
        }

        if average_distances:
            # calculate the average of the distances and return
            mean_distance = sum_of_distances / len(distance_by_words)
            return mean_distance, sorted_distances_by_word

        return sum_of_distances, sorted_distances_by_word

    def run_query(
        self,
        query: Query,
        word_embedding: WordEmbeddingModel,
        distance_type: str = "norm",
        average_distances: bool = True,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Calculate the RND metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to be tested.

        word_embedding_model : WordEmbeddingModel
            An object containing a word embeddings model.

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

        normalize : bool, optional
            True indicates that embeddings will be normalized, by default False

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

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {
                "query_name": query.query_name,
                "result": np.nan,
                "rnd": np.nan,
                "distances_by_word": {},
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        target_0_embeddings = np.array(list(target_embeddings[0].values()))
        target_1_embeddings = np.array(list(target_embeddings[1].values()))
        attribute_0_embeddings = np.array(list(attribute_embeddings[0].values()))
        attribute_0_words = np.array(list(attribute_embeddings[0].keys()))

        distance, distances_by_word = self.__calc_rnd(
            target_0_embeddings,
            target_1_embeddings,
            attribute_0_embeddings,
            attribute_0_words,
            distance_type,
            average_distances,
        )

        return {
            "query_name": query.query_name,
            "result": distance,
            "rnd": distance,
            "distances_by_word": distances_by_word,
        }
