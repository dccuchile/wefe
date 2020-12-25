from typing import Any, Dict, Union
import numpy as np
from ..query import Query
from ..word_embedding_model import PreprocessorArgs, WordEmbeddingModel
from .base_metric import BaseMetric


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

    metric_template = (1, 'n')
    metric_name = 'Mean Average Cosine Similarity'
    metric_short_name = 'MAC'

    def calc_s(self, t, A_j):
        def calc_cos_dist(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return 1 / len(A_j) * np.sum([calc_cos_dist(t, a) for a in A_j])

    def calc_mac(self, T, A):
        first_term = 1 / (len(T))
        mac = first_term * np.sum(
            [np.sum([self.calc_s(t_i, A_j) for A_j in A]) for t_i in T])

        return mac

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
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
        """Calculate the MAC metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets
            for be tested.

        word_embedding : 
            A  object that contain certain word embedding
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
            and a dictionary with the distances of each attribute word
            with respect to the target sets means.
        """
        # checks the types of the provided arguments (only the defaults).
        super().run_query(query, word_embedding, lost_vocabulary_threshold,
                          preprocessor_args, secondary_preprocessor_args,
                          warn_not_found_words, *args, **kwargs)

        # transforming query words into embeddings
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
                'query_name': query.query_name,
                'result': np.nan,
                "rnd": np.nan,
                "distances_by_word": {}
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

        return {'query_name': query.query_name, 'result': result}
