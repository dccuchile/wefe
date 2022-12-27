"""Word Embedding Assosiation Test (WEAT) metric implementation."""
import logging
import math
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import EmbeddingDict, WordEmbeddingModel


class WEAT(BaseMetric):
    r"""Word Embedding Association Test (WEAT).

    The following description of the metric is WEFE's adaptation of what was presented
    in the original WEAT work "Semantics derived automatically from language corpora
    contain human-like biases" [1].

    WEAT receives two sets :math:`T_1` and :math:`T_2` of target words,
    and two sets :math:`A_1` and :math:`A_2` of attribute words and performs a
    hypothesis test on the following null hypothesis:
    There is no difference between the two sets of target words in terms of their
    relative similarity to the similarity with the two sets of attribute words.

    In formal terms, let :math:`T_1` and :math:`T_2` be two sets of target words of
    equal size, and :math:`A_1`, :math:`A_2` the two sets of attribute words.
    Let :math:`\cos(\vec{a}, \vec{b})` denote the cosine of the angle between the
    vectors :math:`\vec{a}` and :math:`\vec{b}`. The test statistic is:

    .. math::

        \text{WEAT}(T_1,T_2,A_1,A_2) = \sum_{x \in T_1} s(x, A_1, A_2) -
        \sum_{y \in T_2} s(y, A_1, A_2)

    where

    .. math::

        s(w, A, B)=\text{mean}_{a \in A} \cos(\vec{w}, \vec{a}) -
        \text{mean}_{b \in B} \cos(\vec{w},\vec{b})

    :math:`s(w,A,B)` measures the association of :math:`w` with the
    attributes, and :math:`\text{WEAT}(T_1,T_2,A_1,A_2)` measures the differential
    association of the two sets of target words with the attribute.

    This metric also contains a variant: WEAT Effect Size (WEAT-ES). This variant
    represents a normalized measure that quantifies how far apart the two distributions
    of association between targets and attributes are. In practical terms, WEAT
    Effect Size makes the metric not dependent on the number of words used in each set.

    .. math::

        \text{WEAT-ES}(T_1,T_2,A_1,A_2) = \frac{\text{mean}_{x \in T_1}\,
        s(x, A_1, A_2) - \text{mean}_{y \in T_2}\, s(y, A_1, A_2) }
        {\text{std-dev}_{w \in T_1 \cup T_2}\, s(w, A_1, A_2)}

    The permutation test measures the (un)likelihood of the null hypothesis by
    computing the probability that a random permutation of the attribute words would
    produce the observed (or greater) difference in sample mean.

    Let :math:`{(T_{1_i},T_{2_i})}_{i}` denote all the partitions of
    :math:`T_1 \cup T_2` into two sets of equal size. The one-sided p-value of the
    permutation test is:

    .. math::

        \text{Pr}_{i}[s(T_{1_i}, T_{2_i}, A_1, A_2) > s(T_1, T_2, A_1, A_2)]

    References
    ----------
    | [1]: Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan.
    |      Semantics derived automatically from language corpora contain human-like
           biases.
    |      Science, 356(6334):183–186, 2017.
    """

    metric_template = (2, 2)
    metric_name = "Word Embedding Association Test"
    metric_short_name = "WEAT"

    def _calc_s(self, w, A, B) -> np.number:

        A_mean_sim = np.mean(cosine_similarity([w], A), dtype=np.float64)
        B_mean_sim = np.mean(cosine_similarity([w], B), dtype=np.float64)
        return A_mean_sim - B_mean_sim

    def _calc_weat(self, X, Y, A, B) -> np.number:
        first_term = np.sum([self._calc_s(x, A, B) for x in X], dtype=np.float64)
        second_term = np.sum([self._calc_s(y, A, B) for y in Y], dtype=np.float64)
        return first_term - second_term

    def _calc_effect_size(self, X, Y, A, B) -> np.number:
        first_term = np.mean([self._calc_s(x, A, B) for x in X], dtype=np.float64)
        second_term = np.mean([self._calc_s(y, A, B) for y in Y], dtype=np.float64)

        std_dev = np.std(
            [self._calc_s(w, A, B) for w in np.concatenate((X, Y))], dtype=np.float64
        )

        return (first_term - second_term) / std_dev

    def _calc_p_value(
        self,
        target_embeddings: List[EmbeddingDict],
        attribute_embeddings: List[EmbeddingDict],
        original_score: np.number,
        iterations: int,
        method: str,
        test_type: str,
        verbose: bool,
    ):
        # TODO: Add joblib or other library to parallelize this function.
        # TODO: Test this function
        # TODO: Implement exact and bootstrap methods
        # TODO: Refactor this function to be able to work with other metrics.
        # maybe this function could be extended from the basemetric class.

        if verbose:
            logging.info(f"weat_original_result {original_score}")

        if method == "exact":
            raise NotImplementedError
        if method != "approximate":
            raise Exception(
                f'p value method should be "exact", "approximate"' f", got {method}."
            )

        # Choose the type of test to be calculated.
        if test_type == "left-sided":

            def test_function(calculated, original):
                return calculated < original

        elif test_type == "right-sided":

            def test_function(calculated, original):
                return calculated > original

        elif test_type == "two-sided":

            def test_function(calculated, original):
                return np.abs(calculated) > original

        else:
            raise Exception(
                f'p value test type should be "left-sided", "right-sided" '
                f'or "two-sided", got {test_type}'
            )

        if not isinstance(iterations, (int, float)):
            raise TypeError(
                f"p value iterations should be int instance, " f"got {iterations}."
            )

        if not isinstance(verbose, bool):
            raise TypeError(f"verbose should be bool instance, got {verbose}.")

        # get attribute embeddings
        attribute_0 = list(attribute_embeddings[0].values())
        attribute_1 = list(attribute_embeddings[1].values())

        # generate the pool of target and attribute embeddings.
        target_0_dict, target_1_dict = target_embeddings
        pool_target_sets = {**target_0_dict, **target_1_dict}
        pool_target_words: List[str] = list(pool_target_sets.keys())

        len_target_0 = len(target_0_dict)
        number_of_target_words = len(pool_target_words)

        total_permutations = math.factorial(number_of_target_words)
        runs = np.min((iterations, total_permutations))

        permutations_seen: Set[Tuple] = set()

        if verbose:
            logging.info(
                f"Number of possible total permutations: {total_permutations}. "
                f"Maximum iterations allowed: {runs}"
            )

        count_pass_function = 0

        while len(permutations_seen) < runs:
            if verbose and len(permutations_seen) % 500 == 0:
                logging.info(f"WEAT p-value: {len(permutations_seen)} / {runs} runs")

            permutation = tuple(
                np.random.choice(
                    pool_target_words, size=number_of_target_words
                ).tolist()
            )
            if permutation not in permutations_seen:
                X_i_words = list(permutation[0:len_target_0])
                Y_i_words = list(permutation[len_target_0:])

                X_i = np.array([pool_target_sets[word] for word in X_i_words])
                Y_i = np.array([pool_target_sets[word] for word in Y_i_words])

                curr_permutation_weat = self._calc_weat(
                    X_i, Y_i, attribute_0, attribute_1
                )

                if test_function(curr_permutation_weat, original_score):
                    count_pass_function += 1

                permutations_seen.add(permutation)

        p_value = (count_pass_function + 1) / (
            runs + 1
        )  # to make p-value unbiased estimator

        if verbose:
            logging.info(
                f"Number of runs: {runs}, Permutations that pass the test function "
                f"type: {count_pass_function}, p-value: {p_value}"
            )
        return p_value

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        return_effect_size: bool = False,
        calculate_p_value: bool = False,
        p_value_test_type: str = "right-sided",
        p_value_method: str = "approximate",
        p_value_iterations: int = 10000,
        p_value_verbose: bool = False,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate the WEAT metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute sets to be tested.

        model : WordEmbeddingModel
            A word embedding model.

        return_effect_size : bool, optional
            Specifies if the returned score in 'result' field of results dict
            is by default WEAT effect size metric, by default False

        calculate_p_value : bool, optional
            Specifies whether the p-value will be calculated through a permutation test.
            Warning: This can increase the computing time quite a lot, by default False.

        p_value_test_type : {'left-sided', 'right-sided', 'two-sided}, optional
            When calculating the p-value, specify the type of test to be performed.
            The options are 'left-sided', 'right-sided' and 'two-sided'
            , by default 'right-sided'

        p_value_method : {'exact', 'approximate'}, optional
            When calculating the p-value, specify the method for calculating the
            p-value. This can be 'exact' and 'approximate'.
            by default 'approximate'.

        p_value_iterations : int, optional
            If the p-value is calculated and the chosen method is 'approximate',
            it specifies the number of iterations that will be performed
            , by default 10000.

        p_value_verbose : bool, optional
            In case of calculating the p-value, specify if notification messages
            will be logged during its calculation, by default False.

        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is
            allowed to lose when transforming its words into embeddings.
            In the case that any set of the query loses proportionally more words
            than this limit, the result values will be np.nan, by default 0.2

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
                specified. True uses 'unicode' by default.
            *   ``preprocessor``: ``Callable``. It receives a function that operates
                on each word. In the case of specifying a function, it overrides the
                default preprocessor (i.e., the previous options stop working).

            A list of preprocessor options allows you to search for several
            variants of the words into the model. For example, the preprocessors
            ``[{}, {"lowercase": True, "strip_accents": True}]``
            ``{}`` allows searching first for the original words in the vocabulary of
            the model. In case some of them are not found,
            ``{"lowercase": True, "strip_accents": True}`` is executed on these words
            and then they are searched in the model vocabulary.

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. 'all' will include all
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
            and the scores of WEAT and the effect size of the metric.

        Examples
        --------
        The following example shows how to run a query that measures gender
        bias using WEAT:

        >>> from wefe.query import Query
        >>> from wefe.utils import load_test_model
        >>> from wefe.metrics import WEAT
        >>>
        >>> # define the query
        >>> query = Query(
        ...     target_sets=[
        ...         ["female", "woman", "girl", "sister", "she", "her", "hers",
        ...          "daughter"],
        ...         ["male", "man", "boy", "brother", "he", "him", "his", "son"],
        ...     ],
        ...     attribute_sets=[
        ...         ["home", "parents", "children", "family", "cousins", "marriage",
        ...          "wedding", "relatives",
        ...         ],
        ...         ["executive", "management", "professional", "corporation", "salary",
        ...          "office", "business", "career",
        ...         ],
        ...     ],
        ...     target_sets_names=["Female terms", "Male Terms"],
        ...     attribute_sets_names=["Family", "Career"],
        ... )
        >>>
        >>> # load the model (in this case, the test model included in wefe)
        >>> model = load_test_model()
        >>>
        >>> # instance the metric and run the query
        >>> WEAT().run_query(query, model)
        {'query_name': 'Female terms and Male Terms wrt Family and Career',
        'result': 0.4634388245467562,
        'weat': 0.4634388245467562,
        'effect_size': 0.45076532408312986,
        'p_value': nan}

        If you want to return the effect size as result value, use
        `return_effect_size` parameter as `True` while running the query.

        >>> WEAT().run_query(query, model, return_effect_size=True)
        {'query_name': 'Female terms and Male Terms wrt Family and Career',
        'result': 0.45076532408312986,
        'weat': 0.4634388245467562,
        'effect_size': 0.45076532408312986,
        'p_value': nan}

        If you want the embeddings to be normalized before calculating the metrics
        use the `normalize` parameter as `True` before executing the query.

        >>> WEAT().run_query(query, model, normalize=True)
        {'query_name': 'Female terms and Male Terms wrt Family and Career',
        'result': 0.4634388248814503,
        'weat': 0.4634388248814503,
        'effect_size': 0.4507653062895615,
        'p_value': nan}

        Using the `calculate_p_value` parameter as `True` you can indicate WEAT to run
        the permutation test and return its p-value. The argument
        `p_value_method='approximate'` indicates that the calculation of the
        permutation test will be approximate, i.e., not all possible permutations
        will be generated.
        Instead, random permutations of the attributes to test will be generated.
        On the other hand, the argument `p_value_iterations`
        indicates the number of permutations that will be generated and tested.

        >>> WEAT().run_query(
        ...     query,
        ...     model,
        ...     calculate_p_value=True,
        ...     p_value_method="approximate",
        ...     p_value_iterations=10000,
        ... )
        {
            'query_name': 'Female terms and Male Terms wrt Family and Career',
            'result': 0.46343879750929773,
            'weat': 0.46343879750929773,
            'effect_size': 0.4507652708557911,
            'p_value': 0.1865813418658134
        }

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
                "weat": np.nan,
                "effect_size": np.nan,
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        target_0 = list(target_embeddings[0].values())
        target_1 = list(target_embeddings[1].values())
        attribute_0 = list(attribute_embeddings[0].values())
        attribute_1 = list(attribute_embeddings[1].values())

        # if the requested value is the effect size:
        weat_effect_size = self._calc_effect_size(
            target_0, target_1, attribute_0, attribute_1
        )
        weat = self._calc_weat(target_0, target_1, attribute_0, attribute_1)

        if calculate_p_value:
            p_value = self._calc_p_value(
                target_embeddings=target_embeddings,
                attribute_embeddings=attribute_embeddings,
                original_score=weat,
                test_type=p_value_test_type,
                method=p_value_method,
                iterations=p_value_iterations,
                verbose=p_value_verbose,
            )
        else:
            p_value = np.nan

        # return in result field effect_size
        if return_effect_size:
            return {
                "query_name": query.query_name,
                "result": weat_effect_size,
                "weat": weat,
                "effect_size": weat_effect_size,
                "p_value": p_value,
            }

        # return in result field weat
        return {
            "query_name": query.query_name,
            "result": weat,
            "weat": weat,
            "effect_size": weat_effect_size,
            "p_value": p_value,
        }
