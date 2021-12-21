"""Relational Inner Product Association Test."""
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class RIPA(BaseMetric):
    """An implementation of the Relational Inner Product Association Test, proposed by [1][2].

    RIPA is most interpretable with a single pair of target words, although
    this function returns the values for every attribute averaged across all base pairs.

    NOTE: As the variance tends to be high depending on the base pair chosen, it is
    recommended that only a single pair of target words is used as input to the
    function.

    This metric follows the following steps:
    
    1. The input is the word vectors for a pair of target word sets, and an attribute set.
       Example: Target Set A (Masculine), Target Set B (Feminine), Attribute Set
       (Career).
    2. Calculate the difference between the word vector of a pair of target set words.
    3. Calculate the dot product between this difference and the attribute word vector.
    4. Return the average RIPA score across all attribute words, and the average RIPA
       score for each target pair for an attribute set.

    References
    ----------
    | [1]: Ethayarajh, K., & Duvenaud, D., & Hirst, G. (2019, July). Understanding Undesirable Word Embedding Associations.
    | [2]: https://kawine.github.io/assets/acl2019_bias_slides.pdf
    | [3]: https://kawine.github.io/blog/nlp/2019/09/23/bias.html

    """

    # replace with the parameters of your metric
    metric_template = (2, 1)
    metric_name = "Relational Inner Product Association"
    metric_short_name = "RIPA"

    def _b_vec(self, word1, word2):
        # calculating the relation vector
        vec = np.array(word1) - np.array(word2)
        norm = np.linalg.norm(vec)
        return vec / norm

    def _ripa_calc(self, word_vec, bvec):
        # calculating the dot product of the relation vector with the attribute
        # word vector
        return np.dot(word_vec, bvec)

    def _calc_metric(
        self, target_embeddings, attribute_embeddings
    ) -> Tuple[np.floating, Dict[str, Dict[str, np.floating]]]:
        """Calculate the metric.

        Parameters
        ----------
        target_embeddings : np.array
            An array with dicts. Each dict represents an target set.
            A dict is composed with a word and its embedding as key, value respectively.
        attribute_embeddings : np.array
            An array with dicts. Each dict represents an attribute set.
            A dict is composed with a word and its embedding as key, value respectively.

        Returns
        -------
        np.float
            The value of the calculated metric, averaged across all the RIPA scores of
            the attributes.
        dict
            The mean value ± the standard deviation (across all target pairs) of the
            attribute word's RIPA score.
        """
        # word vectors from the embedding model for all the words in each of the
        # target sets
        target_embeddings_0 = list(target_embeddings[0].values())
        target_embeddings_1 = list(target_embeddings[1].values())

        # word vectors from the embedding model for all the words in the attribute set
        attribute_embeddings_0 = np.array(list(attribute_embeddings[0].values()))

        target_length = len(target_embeddings_0)  # length of the target set
        attributes = list(
            attribute_embeddings[0].keys()
        )  # list of all the attribute words

        ripa_scores: Dict[str, list] = {}

        ripa_oa_mean = []
        ripa_oa_std = []

        # calculating the ripa score for each attribute word with each target pair
        for word in range(len(attribute_embeddings_0)):
            ripa_scores[attributes[word]] = []
            for index in range(target_length - 1):
                bvec = self._b_vec(
                    target_embeddings_0[index], target_embeddings_1[index]
                )
                score = self._ripa_calc(attribute_embeddings_0[word], bvec)
                ripa_scores[attributes[word]].append(score)

        # calculating the mean of the ripa score across all target pairs for every
        # attribute word
        for rip in attributes:
            ripa_oa_mean.append(np.mean(ripa_scores[rip]))
            ripa_oa_std.append(np.std(ripa_scores[rip]))

        # creating a dictionary with the direction of the RIPA scores for every
        # corresponding attribute word
        word_values = {}
        for i in range(len(ripa_oa_mean)):
            word_values[attributes[i]] = {
                "mean": ripa_oa_mean[i],
                "std": ripa_oa_std[i],
            }

        # Returning the mean of the RIPA scores for every attribute word, and the
        # dictionary with the RIPA scores for every corresponding attribute word.
        return np.mean(ripa_oa_mean), word_values

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
        """Calculate the Example Metric metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute sets to be tested.

        model : WordEmbeddingModel
            A word embedding model.

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
            and other scores.

        Examples
        --------
        >>> from wefe.metrics import RIPA
        >>> from wefe.query import Query
        >>> from wefe.utils import load_test_model
        >>> 
        >>> # define the query
        ... query = Query(
        ...     target_sets=[
        ...         ["female", "woman", "girl", "sister", "she", "her", "hers",
        ...         "daughter"],
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
        >>> RIPA().run_query(query, model) # doctest: +SKIP
        {
            'query_name': 'Female terms and Male Terms wrt Family',
            'result': 0.18600442,
            'ripa': 0.18600442,
            'word_values': {
                'home': {'mean': 0.008022693, 'std': 0.07485135},
                'parents': {'mean': 0.2038254, 'std': 0.30801567},
                'children': {'mean': 0.30370313, 'std': 0.2787335},
                'family': {'mean': 0.07277942, 'std': 0.20808344},
                'cousins': {'mean': -0.040398445, 'std': 0.27916202},
                'marriage': {'mean': 0.37971354, 'std': 0.31494072},
                'wedding': {'mean': 0.39682359, 'std': 0.28507116},
                'relatives': {'mean': 0.16356598, 'std': 0.21053126}
            }
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
                "ripa": np.nan,
                "word_values": np.nan,
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        result, word_values = self._calc_metric(target_embeddings, attribute_embeddings)

        # return the results.
        return {
            "query_name": query.query_name,
            "result": result,
            "ripa": result,
            "word_values": word_values,
        }
