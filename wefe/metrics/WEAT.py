from typing import Any, Dict, Union
import numpy as np

from .base_metric import BaseMetric
from ..word_embedding import WordEmbedding
from ..query import Query


class WEAT(BaseMetric):
    """A implementation of Word Embedding Association Test (WEAT).
    
    It measures the degree of association between two sets of target words and 
    two sets of attribute words through a permutation test.  
    
    References
    ----------
    Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. 
    Semantics derived automatically from language corpora contain human-like biases.
    Science,356(6334):183–186, 2017.
    """
    def __init__(self):
        super().__init__((2, 2), 'Word Embedding Association Test', 'WEAT')

    def __calc_s(self, w, A, B):
        def calc_cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        A_cosine_distance = np.array([calc_cos_sim(w, a) for a in A])
        B_cosine_distance = np.array([calc_cos_sim(w, b) for b in B])
        return np.mean(A_cosine_distance) - np.mean(B_cosine_distance)

    def __calc_effect_size(self, X, Y, A, B):
        first_term = np.mean(np.array([self.__calc_s(x, A, B) for x in X]))
        second_term = np.mean(np.array([self.__calc_s(y, A, B) for y in Y]))

        std_dev = np.std(
            np.array([self.__calc_s(w, A, B) for w in np.concatenate((X, Y))]))

        return (first_term - second_term) / std_dev

    def __calc_weat(self, X, Y, A, B):
        first_term = np.array([self.__calc_s(x, A, B) for x in X])
        second_term = np.array([self.__calc_s(y, A, B) for y in Y])
        return np.sum(first_term) - np.sum(second_term)

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbedding,
                  return_effect_size: bool = False,
                  lost_vocabulary_threshold: float = 0.2,
                  preprocessor_options: Dict = {
                      'strip_accents': False,
                      'lowercase': False,
                      'preprocessor': None,
                  },
                  secondary_preprocessor_options: Union[Dict, None] = None,
                  warn_not_found_words: bool = False,
                  *args: Any,
                  **kwargs: Any) -> Dict[str, Any]:
        """Calculate the WEAT metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to 
            be tested.

        word_embedding : WordEmbedding
            A WordEmbedding object that contains certain word embedding 
            pretrained model.

        return_effect_size : bool, optional
            Specifies if the returned score in 'result' field of results dict 
            is by default WEAT effect size metric, by default False
        
        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is 
            allowed to lose when transforming its words into embeddings. 
            In the case that any set of the query loses proportionally more words 
            than this limit, the result values will be np.nan, by default 0.2
        
        preprocessor_options : Dict, optional
            Dictionary with options for pre-processing words, by default {}
            The options for the dict are: 
            - lowercase: bool. Indicates if the words are transformed to lowercase.
            - strip_accents: bool, {'ascii', 'unicode'}: Specifies if the accents of 
                             the words are eliminated. The stripping type can be 
                             specified. True uses 'unicode' by default.
            - preprocessor: Callable. It receives a function that operates on each 
                            word. In the case of specifying a function, it overrides 
                            the default preprocessor (i.e., the previous options 
                            stop working).
            , by default { 'strip_accents': False, 'lowercase': False, 'preprocessor': None, }
        
        secondary_preprocessor_options : Union[Dict, None], optional
            Dictionary with options for pre-processing words (same as the previous 
            parameter), by default None.
            Indicates that in case a word is not found in the model's vocabulary 
            (using the default preprocessor or specified in preprocessor_options), 
            the function performs a second search for that word using the preprocessor 
            specified in this parameter, by default None

        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary
            , by default False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name, the resulting score of the metric, 
            and the scores of WEAT and the effect size of the metric.
        """
        # checks the types of the provided arguments (only the defaults).
        super().run_query(query, word_embedding, lost_vocabulary_threshold,
                          preprocessor_options, secondary_preprocessor_options,
                          warn_not_found_words, *args, **kwargs)

        # transforming query words into embeddings
        embeddings = word_embedding.get_embeddings_from_query(
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessor_options=preprocessor_options,
            secondary_preprocessor_options=secondary_preprocessor_options,
            warn_not_found_words=warn_not_found_words)

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {
                'query_name': query.query_name,
                'result': np.nan,
                'weat': np.nan,
                'effect_size': np.nan
            }

        # get the target and attribute embeddings
        target_embeddings = embeddings['target_embeddings']
        attribute_embeddings = embeddings['attribute_embeddings']

        target_0 = list(target_embeddings[0].values())
        target_1 = list(target_embeddings[1].values())
        attribute_0 = list(attribute_embeddings[0].values())
        attribute_1 = list(attribute_embeddings[1].values())

        # if the requested value is the effect size:

        weat_effect_size = self.__calc_effect_size(target_0, target_1, attribute_0,
                                                   attribute_1)
        weat = self.__calc_weat(target_0, target_1, attribute_0, attribute_1)

        # return in result field effect_size
        if return_effect_size:
            return {
                'query_name': query.query_name,
                'result': weat_effect_size,
                'weat': weat,
                'effect_size': weat_effect_size
            }

        # return in result field weat
        return {
            'query_name': query.query_name,
            'result': weat,
            'weat': weat,
            'effect_size': weat_effect_size
        }
