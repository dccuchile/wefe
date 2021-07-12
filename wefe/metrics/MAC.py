"""Mean Average Cosine Similarity (MAC) implementation."""
from typing import Any, Callable, Dict, List, Union
import numpy as np
from scipy.spatial import distance

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.preprocessing import get_embeddings_from_query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import cosine_distance


class MAC(BaseMetric):
    """Mean Average Cosine Similarity (MAC) .

    The general steps of the test are as follows [1]:

    1. Embedd all target and attribute words.
    2. For each target set:
        * For each word embedding in the target set:
            * For each attribute set:
                * Calculate the cosine similarity of the target embedding and
                each attribute embedding of the set.

                * Calculate the mean of the cosines similarities and save it in a array.
    3. Average all the mean cosine similarities and return the calculated score.

    References
    ----------

    | [1]: Thomas Manzini, Lim Yao Chong,Alan W Black, and Yulia Tsvetkov.
    | Black is to criminalas caucasian is to police: Detecting and removing multiclass
    | bias in word embeddings.
    | In Proceedings of the 2019 Conference of the North American Chapter of the
    | Association for Computational Linguistics:
    | Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621,
    | Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
    | [2]: https://github.com/TManzini/DebiasMulticlassWordEmbedding/blob/master/Debiasing/evalBias.py
    """

    metric_template = ("n", "n")
    metric_name = "Mean Average Cosine Similarity"
    metric_short_name = "MAC"

    def _calc_s(self, t: np.ndarray, A_j: np.ndarray) -> float:
        """Calculate the mean cos similarity of a target embedding and a attribute set.

        Parameters
        ----------
        t : np.ndarray
            A target embedding
        A_j : np.ndarray
            An attribute embedding set.

        Returns
        -------
        float
            The mean cosine similarity between the target embedding and the attribute
            set calculated.
        """
        return np.mean([distance.cosine(t, a_i) for a_i in A_j])

    def _calc_mac(self, T, A):

        # dict that will store the s scores by target word and attribute set.
        targets_eval = {}
        # list that will store the s scores
        targets_eval_scores = []

        # T_i -> Current target set
        # t_i -> Current target embedding
        # A_j -> Current attribute set

        # for each target set
        for T_i_name, T_i_vecs in T.items():
            targets_eval[T_i_name] = {}
            # for each embedding in the current target set
            for t_i_word, t_i_vec in T_i_vecs.items():
                targets_eval[T_i_name][t_i_word] = {}
                # for each attribute set
                for A_j_name, A_j_vecs in A.items():
                    # calculate s score
                    score = self._calc_s(t_i_vec, A_j_vecs.values())
                    # add the score to the variables that will store it
                    targets_eval[T_i_name][t_i_word][A_j_name] = score
                    targets_eval_scores.append(score)

        # obtain mac by calculating the mean over the s scores in targets_eval_scores.
        mac_score = np.mean(np.array(targets_eval_scores))

        return mac_score, targets_eval

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
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

        model : WordEmbeddingModel
            A word embedding model.

        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is
            allowed to lose when transforming its words into embeddings.
            In the case that any set of the query loses proportionally more words
            than this limit, the result values will be np.nan, by default 0.2

        preprocessors : List[Dict[str, Union[str, bool, Callable]]]
            A list with preprocessor options.

            A dictionary of preprocessing options is a dictionary that specifies what
            transformations will be made to each word prior to being searched in the
            word embedding model vocabulary.
            For example, `{'lowecase': True, 'strip_accents': True}` allows you to
            transform the words to lowercase and remove the accents and then search
            for them in the model.
            Note that an empty dictionary `{}` indicates that no transformation
            will be made to any word.

            A list of these preprocessor options will allow you to search for several
            variants of the words (depending on the search strategy) into the model.
            For example `[{}, {'lowecase': True, 'strip_accents': True}]` allows you
            to search for each word, first, without any transformation and then,
            transformed to lowercase and without accents.

            The available word preprocessing options are as follows (it is not necessary
            to put them all):

            - `lowercase`: `bool`. Indicates if the words are transformed to lowercase.
            - `uppercase`: `bool`. Indicates if the words are transformed to uppercase.
            - `titlecase`: `bool`. Indicates if the words are transformed to titlecase.
            - `strip_accents`: `bool`, `{'ascii', 'unicode'}`: Specifies if the accents
                                of the words are eliminated. The stripping type can be
                                specified. True uses 'unicode' by default.
            - `preprocessor`: `Callable`. It receives a function that operates on each
                            word. In the case of specifying a function, it overrides
                            the default preprocessor (i.e., the previous options
                            stop working).
            by default [{}].

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. all' will include all
            transformed words found, by default "first".

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
        self._check_input(query, model)

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
                "mac": np.nan,
                "targets_eval": None,
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        mac, targets_eval = self._calc_mac(target_sets, attribute_sets)

        return {
            "query_name": query.query_name,
            "result": mac,
            "mac": mac,
            "targets_eval": targets_eval,
        }
