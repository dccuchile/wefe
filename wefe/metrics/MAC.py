from typing import Any, Callable, Dict, List, Union
import numpy as np


from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.preprocessing import get_embeddings_from_query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import cosine_distance


class MAC(BaseMetric):
    """A implementation of Mean Average Cosine Similarity (MAC).

    References
    -------
        [1] Thomas Manzini, Lim Yao Chong,Alan W Black, and Yulia Tsvetkov.
        Black is to criminalas caucasian is to police: Detecting and removing multiclass
        bias in word embeddings.
        In Proceedings of the 2019 Conference of the North American Chapter of the
        Association for Computational Linguistics:
        Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621,
        Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
    """

    metric_template = (1, "n")
    metric_name = "Mean Average Cosine Similarity"
    metric_short_name = "MAC"

    def calc_s(self, t, A_j):
        # def calc_cos_dist(a, b):
        #     return 1 - cosine_similarity(a, b)
        # return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return 1 / len(A_j) * np.sum([cosine_distance(t, a) for a in A_j])

    def calc_mac(self, T, A):
        first_term = 1 / (len(T))
        mac = first_term * np.sum(
            [np.sum([self.calc_s(t_i, A_j) for A_j in A]) for t_i in T]
        )

        return mac

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
        """Calculate the MAC metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets
            for be tested.

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
            embeddings model. For example, `{'lowecase': True, 'strip_accents': True}` will
            allow you to search for words in the word_set transformed to lowercase and
            without accents.
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

        # get the target and attribute embeddings dicts
        target_0_embeddings = list(target_embeddings[0].values())
        attribute_embeddings_all_sets = [
            list(attibute_dict.values()) for attibute_dict in attribute_embeddings
        ]

        result = self.calc_mac(target_0_embeddings, attribute_embeddings_all_sets)

        return {"query_name": query.query_name, "result": result}
