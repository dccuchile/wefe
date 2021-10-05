"""An example of how to implement metrics in WEFE."""

from typing import Any, Callable, Dict, List, Union

import numpy as np
from scipy.spatial import distance
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import EmbeddingDict, WordEmbeddingModel


class ExampleMetric(BaseMetric):
    """Example class intended to guide the implementation of a fairness metric."""

    # replace with the parameters of your metric
    metric_template = (
        2,
        1,
    )  # cardinalities of the targets and attributes sets that your metric will accept.
    metric_name = "Example Metric"
    metric_short_name = "EM"

    def _calc_metric(
        self,
        target_embeddings: List[EmbeddingDict],
        attribute_embeddings: List[EmbeddingDict],
    ) -> float:
        """Calculate the metric.

        Parameters
        ----------
        target_embeddings : List[EmbeddingDict]
            An array with EmbeddingDict. Each dictionary represents an target set. 
             A dict is composed with a word and its embedding as key, value respectively.
        attribute_embeddings : List[EmbeddingDict]
            [An array with dicts. Each dictionary represents an attribute set. 
             A dict is composed with a word and its embedding as key, value respectively.

        Returns
        -------
        np.float
            The value of the calculated metric.
        """

        # get the embeddings from the dicts
        target_embeddings_0 = np.array(list(target_embeddings[0].values()))
        target_embeddings_1 = np.array(list(target_embeddings[1].values()))

        attribute_embeddings_0 = np.array(list(attribute_embeddings[0].values()))

        # calculate the average embedding by target and attribute set.
        target_embeddings_0_avg = np.mean(target_embeddings_0, axis=0)
        target_embeddings_1_avg = np.mean(target_embeddings_1, axis=0)
        attribute_embeddings_0_avg = np.mean(attribute_embeddings_0, axis=0)

        # calculate the distances between the target sets and the attribute set
        dist_target_0_attr = distance.cosine(
            target_embeddings_0_avg, attribute_embeddings_0_avg
        )
        dist_target_1_attr = distance.cosine(
            target_embeddings_1_avg, attribute_embeddings_0_avg
        )

        # subtract the distances
        metric_result = dist_target_0_attr - dist_target_1_attr
        return metric_result

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        # any parameter that you need
        # ...,
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
            A Query object that contains the target and attribute word sets to be tested.

        model : WordEmbeddingModel
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
            and other scores.
        """
        # check the types of the provided arguments (only the defaults).
        self._check_input(query, model, kwargs)

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
                "query_name": query.query_name,  # the name of the evaluated query
                "result": np.nan,  # the result of the metric
                "em": np.nan,  # result of the calculated metric (recommended)
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # commonly, you only will need the embeddings of the sets.
        # this can be obtained by using:
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        # From here, the code can vary quite a bit depending on what you need.
        # metric operations. It is recommended to calculate it in another method(s).

        result = self._calc_metric(target_embeddings, attribute_embeddings)

        # return the results.
        return {"query_name": query.query_name, "result": result, "em": result}
