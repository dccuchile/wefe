"""Relative Norm Distance (RND) metric implementation."""
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class RND(BaseMetric):
    """Relative Norm Distance (RND).

    It measures the relative strength of association of a set of neutral words
    with respect to two groups.

    References
    ----------
    | [1]: Nikhil Garg, Londa Schiebinger, Dan Jurafsky, and James Zou.
    | Word embeddings quantify 100 years of gender and ethnic stereotypes.
    | Proceedings of the National Academy of Sciences, 115(16):E3635–E3644,2018.
    | [2]: https://github.com/nikhgarg/EmbeddingDynamicStereotypes
    """

    metric_template = (2, 1)
    metric_name = "Relative Norm Distance"
    metric_short_name = "RND"

    def __calc_distance(
        self, vec1: np.ndarray, vec2: np.ndarray, distance_type: str = "norm",
    ) -> float:
        if distance_type == "norm":
            return np.linalg.norm(np.subtract(vec1, vec2))
        elif distance_type == "cos":
            # c = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            c = cosine_similarity([vec1], [vec2]).flatten()
            return c[0]
        else:
            raise ValueError(
                'distance_type can be either "norm" or "cos", '
                "got: {} ".format(distance_type)
            )

    def __calc_rnd(
        self,
        target_0: np.ndarray,
        target_1: np.ndarray,
        attribute: np.ndarray,
        attribute_words: list,
        distance_type: str,
    ) -> Tuple[float, Dict[str, float]]:

        # calculates the average wv for the group words.
        target_1_avg_vector = np.average(target_0, axis=0)
        target_2_avg_vector = np.average(target_1, axis=0)

        sum_of_distances = 0.0
        distance_by_words = {}

        for attribute_word_index, attribute_embedding in enumerate(attribute):

            # calculate the distance
            current_distance = self.__calc_distance(
                attribute_embedding, target_1_avg_vector, distance_type=distance_type,
            ) - self.__calc_distance(
                attribute_embedding, target_2_avg_vector, distance_type=distance_type,
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

        # calculate the average of the distances and return
        mean_distance = sum_of_distances / len(distance_by_words)
        return mean_distance, sorted_distances_by_word

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        distance: str = "norm",
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
            A Query object that contains the target and attribute sets to be tested.

        model : WordEmbeddingModel
            A word embedding model.

        distance : str, optional
            Specifies which type of distance will be calculated. It could be:
            {norm, cos} , by default 'norm'.

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
            A dictionary with the query name, the resulting score of the metric,
            and a dictionary with the distances of each attribute word
            with respect to the target sets means.

        Examples
        --------
        The following example shows how to run a query that measures gender
        bias using RND:

        >>> from wefe.metrics import RND
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
        >>>
        >>> # load the model (in this case, the test model included in wefe)
        >>> model = load_test_model()
        >>>
        >>> # instance the metric and run the query
        >>> RND().run_query(query, model) # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family',
         'result': 0.030381828546524048,
         'rnd': 0.030381828546524048,
         'distances_by_word': {'wedding': -0.1056304,
                               'marriage': -0.10163283,
                               'children': -0.068374634,
                               'parents': 0.00097084045,
                               'relatives': 0.0483346,
                               'family': 0.12408042,
                               'cousins': 0.17195654,
                               'home': 0.1733501}}
        >>>

        If you want the embeddings to be normalized before calculating the metrics
        use the normalize parameter as True before executing the query.

        >>> RND().run_query(query, model, normalize=True) # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family',
         'result': -0.006278775632381439,
         'rnd': -0.006278775632381439,
         'distances_by_word': {'children': -0.05244279,
                               'wedding': -0.04642248,
                               'marriage': -0.04268837,
                               'parents': -0.022358716,
                               'relatives': 0.005497098,
                               'family': 0.023389697,
                               'home': 0.04009247,
                               'cousins': 0.044702888}}
        
        
        If you want to use cosine distance instead of euclidean norm
        use the distance parameter as 'cos' before executing the query.

        >>> RND().run_query(query, model, normalize=True, distance='cos') # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family',
         'result': 0.03643466345965862,
         'rnd': 0.03643466345965862,
         'distances_by_word': {'cousins': -0.035989374,
                               'home': -0.026971221,
                               'family': -0.009296179,
                               'relatives': 0.015690982,
                               'parents': 0.051281124,
                               'children': 0.09255883,
                               'marriage': 0.09959312,
                               'wedding': 0.104610026}}
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

        # get a list with the transformed attribute words
        attribute_0_words = list(attribute_embeddings[0].keys())

        rnd, distances_by_word = self.__calc_rnd(
            target_0_embeddings,
            target_1_embeddings,
            attribute_0_embeddings,
            attribute_0_words,
            distance,
        )

        return {
            "query_name": query.query_name,
            "result": rnd,
            "rnd": rnd,
            "distances_by_word": distances_by_word,
        }
