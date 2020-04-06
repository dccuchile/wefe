from .base_metric import BaseMetric
from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from scipy.spatial import distance
import numpy as np


class ExampleMetric(BaseMetric):
    def __init__(self):

        template_needed = (2, 1)
        metric_name = 'Example Metric'
        metric_short_name = 'EM'
        super().__init__(template_needed, metric_name, metric_short_name)

    def __calc_metric(self, target_embeddings, attribute_embeddings):
        """Calculates the metric.
        
        Parameters
        ----------
        target_embeddings : np.array
            An array with dicts. Each dict represent an target set. A dict is composed with a word and its embedding as key, value respectively.
        attribute_embeddings : np.array
            An array with dicts. Each dict represent an attribute set. A dict is composed with a word and its embedding as key, value respectively.
        
        Returns
        -------
        np.float
            The value of the calculated metric.
        """

        # get the embeddings from the dicts
        target_embeddings_0 = np.array(list(target_embeddings[0].values()))
        target_embeddings_1 = np.array(list(target_embeddings[1].values()))

        attribute_embeddings_0 = np.array(
            list(attribute_embeddings[0].values()))

        # calculate the average embedding by target and attribute set.
        target_embeddings_0_avg = np.mean(target_embeddings_0, axis=0)
        target_embeddings_1_avg = np.mean(target_embeddings_1, axis=0)
        attribute_embeddings_0_avg = np.mean(attribute_embeddings_0, axis=0)

        # calculate the distances between the target sets and the attribute set
        dist_target_0_attr = distance.cosine(target_embeddings_0_avg,
                                             attribute_embeddings_0_avg)
        dist_target_1_attr = distance.cosine(target_embeddings_1_avg,
                                             attribute_embeddings_0_avg)

        # substract the distances
        metric_result = dist_target_0_attr - dist_target_1_attr
        return metric_result

    def run_query(self, query: Query, word_embedding: WordEmbeddingModel,
                  lost_vocabulary_threshold: float = 0.2,
                  warn_filtered_words: bool = True):

        # check the inputs
        self._check_input(query, word_embedding, lost_vocabulary_threshold,
                          warn_filtered_words)

        # get the embeddings
        embeddings = self._get_embeddings_from_query(
            query, word_embedding, warn_filtered_words,
            lost_vocabulary_threshold)

        # if there is any/some set has less words than the allowed limit, return the default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # separate the embedding tuple
        target_embeddings, attribute_embeddings = embeddings

        # execute the metric
        metric_result = self.__calc_metric(target_embeddings,
                                           attribute_embeddings)

        return {
            "query_name": query.query_name_,
            "result": metric_result,
        }