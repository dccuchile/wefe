"""Mean Average Cosine Similarity (MAC) implementation."""
from typing import Any, Callable, Dict, List, Union

import numpy as np
from scipy.spatial import distance
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class MAC(BaseMetric):
    """Mean Average Cosine Similarity (MAC).

    The general steps of the test are as follows [1].

    1. Embed all target and attribute words.
    2. For each target set:

        * For each word embedding in the target set:

            * For each attribute set:

                * Calculate the cosine similarity of the target embedding and
                each attribute embedding of the set.

                * Calculate the mean of the cosines similarities and save it in a array.

    3. Average all the mean cosine similarities and return the calculated score.


    The closer the value is to 1, the less biased the query will be.

    References
    ----------
    | [1]: Thomas Manzini, Lim Yao Chong,Alan W Black, and Yulia Tsvetkov.
      Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass
      Bias in Word Embeddings.
      In Proceedings of the 2019 Conference of the North American Chapter of the
      Association for Computational Linguistics:
      Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621,
      Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
    | [2]: https://github.com/TManzini/DebiasMulticlassWordEmbedding/blob/master/Debiasing/evalBias.py
    """

    metric_template = ("n", "n")
    metric_name = "Mean Average Cosine Similarity"
    metric_short_name = "MAC"

    def _calc_s(self, t: np.ndarray, A_j: np.ndarray) -> np.number:
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
        >>> from wefe.metrics import MAC
        >>> from wefe.query import Query
        >>> from wefe.utils import load_test_model
        >>>
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
        ...     attribute_sets_names=["Family", "Careers"],
        ... )
        >>>
        >>> # load the model (in this case, the test model included in wefe)
        >>> model = load_test_model()
        >>>
        >>> # instance the metric and run the query
        >>> MAC().run_query(query, model) # doctest: +SKIP
        {'query_name': 'Female terms and Male Terms wrt Family and Careers',
        'result': 0.8416415235615204,
        'mac': 0.8416415235615204,
        'targets_eval': {'Female terms': {'female': {'Family': 0.9185737599618733,
            'Careers': 0.916069650076679},
            'woman': {'Family': 0.752434104681015, 'Careers': 0.9377805145923048},
            'girl': {'Family': 0.707457959651947, 'Careers': 0.9867974997032434},
            'sister': {'Family': 0.5973392464220524, 'Careers': 0.9482253392925486},
            'she': {'Family': 0.7872791914269328, 'Careers': 0.9161583095556125},
            'her': {'Family': 0.7883057091385126, 'Careers': 0.9237247597193345},
            'hers': {'Family': 0.7385367527604103, 'Careers': 0.9480051446007565},
            'daughter': {'Family': 0.5472579970955849, 'Careers': 0.9277344475267455}},
        'Male Terms': {'male': {'Family': 0.8735092766582966,
            'Careers': 0.9468009045813233},
            'man': {'Family': 0.8249392118304968, 'Careers': 0.9350165261421353},
            'boy': {'Family': 0.7106057899072766, 'Careers': 0.9879048476286698},
            'brother': {'Family': 0.6280269809067249, 'Careers': 0.9477180293761194},
            'he': {'Family': 0.8693044614046812, 'Careers': 0.8771287016716087},
            'him': {'Family': 0.8230192996561527, 'Careers': 0.888683641096577},
            'his': {'Family': 0.8876195731572807, 'Careers': 0.8920885202242061},
            'son': {'Family': 0.5764635019004345, 'Careers': 0.9220191016211174}}}}
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
