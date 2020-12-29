"""An example of how to implement metrics in WEFE."""

from typing import Any, Dict, List, Union

import numpy as np
from scipy.spatial import distance

from ..metrics.base_metric import BaseMetric
from ..query import Query
from ..word_embedding_model import EmbeddingDict, WordEmbeddingModel, PreprocessorArgs


class ExampleMetric(BaseMetric):
    """Example class intended to guide the implementation of a fairness metric."""

    # replace with the parameters of your metric
    metric_template = (
        2, 1
    )  # cardinalities of the targets and attributes sets that your metric will accept.
    metric_name = 'Example Metric'
    metric_short_name = 'EM'

    def _calc_metric(self, target_embeddings: List[EmbeddingDict],
                     attribute_embeddings: List[EmbeddingDict]) -> np.float:
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
        dist_target_0_attr = distance.cosine(target_embeddings_0_avg,
                                             attribute_embeddings_0_avg)
        dist_target_1_attr = distance.cosine(target_embeddings_1_avg,
                                             attribute_embeddings_0_avg)

        # subtract the distances
        metric_result = dist_target_0_attr - dist_target_1_attr
        return metric_result

    def run_query(
            self,
            query: Query,
            word_embedding: WordEmbeddingModel,
            # any parameter that you need
            # ...,
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
        """Calculate the Example Metric metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to 
            be tested.

        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contains certain word embedding 
            pretrained model.
        
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
            and other scores.
        """
        # check the types of the provided arguments (only the defaults).
        super().run_query(query, word_embedding, lost_vocabulary_threshold,
                          preprocessor_args, secondary_preprocessor_args,
                          warn_not_found_words, *args, **kwargs)

        # transform query word sets into embeddings
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
                'query_name': query.query_name,  # the name of the evaluated query
                'result': np.nan,  # the result of the metric
                'em': np.nan,  # result of the calculated metric (recommended)
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
        return {"query_name": query.query_name, "result": result, 'em': result}
