import logging
from typing import Any, Dict, List, Set, Tuple, Union
import random
import math

import numpy as np

from .base_metric import BaseMetric
from ..word_embedding_model import EmbeddingDict, EmbeddingSets, PreprocessorArgs, WordEmbeddingModel
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

    metric_template = (2, 2)
    metric_name = 'Word Embedding Association Test'
    metric_short_name = 'WEAT'

    def _calc_s(self, w, A, B):
        def calc_cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        A_cosine_distance = np.array([calc_cos_sim(w, a) for a in A])
        B_cosine_distance = np.array([calc_cos_sim(w, b) for b in B])
        return np.mean(A_cosine_distance) - np.mean(B_cosine_distance)

    def _calc_effect_size(self, X, Y, A, B) -> float:
        first_term = np.mean(np.array([self._calc_s(x, A, B) for x in X]))
        second_term = np.mean(np.array([self._calc_s(y, A, B) for y in Y]))

        std_dev = np.std(
            np.array([self._calc_s(w, A, B) for w in np.concatenate((X, Y))]))

        return (first_term - second_term) / std_dev

    def _calc_weat(self, X, Y, A, B) -> float:
        first_term = np.array([self._calc_s(x, A, B) for x in X])
        second_term = np.array([self._calc_s(y, A, B) for y in Y])
        return np.sum(first_term) - np.sum(second_term)

    def _calc_p_value(self, target_embeddings: List[EmbeddingDict],
                      attribute_0: np.ndarray, attribute_1: np.ndarray,
                      weat_original_result: float, max_permutations: int,
                      log_p_value_status: bool):

        if log_p_value_status:
            logging.info(f'weat_original_result {weat_original_result}')

        target_0_dict, target_1_dict = target_embeddings
        target_sets_joined = {**target_0_dict, **target_1_dict}
        target_words_joined: List[str] = list(target_sets_joined.keys())

        number_of_target_words = len(target_words_joined)

        median = int(number_of_target_words / 2)
        total_permutations = math.factorial(number_of_target_words)
        runs = np.min((max_permutations, total_permutations))

        permutations_seen: Set[Tuple] = set()

        if log_p_value_status:
            logging.info(f'Number of possible total permutations: {total_permutations}. '
                         f'Maximum iterations allowed: {runs}')

        count_greater = 0

        while len(permutations_seen) < runs:
            if len(permutations_seen) % 500 == 0 and log_p_value_status:
                logging.info(f'WEAT p-value: {len(permutations_seen)} / {runs} runs')

            permutation = tuple(
                np.random.choice(target_words_joined,
                                 size=number_of_target_words).tolist())
            if permutation not in permutations_seen:
                X_i_words = list(permutation[0:median])
                Y_i_words = list(permutation[median:])

                X_i = np.array([target_sets_joined[word] for word in X_i_words])
                Y_i = np.array([target_sets_joined[word] for word in Y_i_words])

                weat_hat = self._calc_weat(X_i, Y_i, attribute_0, attribute_1)

                if weat_hat > weat_original_result:
                    count_greater += 1

                permutations_seen.add(permutation)

        p_value = count_greater / runs

        if log_p_value_status:
            logging.info(f'Number of runs: {runs}, Permutations with greater WEAT:'
                         f'{count_greater}, p-value: {p_value}')
        return p_value

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  return_effect_size: bool = False,
                  calc_p_value: bool = False,
                  max_p_value_permutations: int = 10000,
                  log_p_value_status: bool = False,
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
        """Calculate the WEAT metric over the provided parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to 
            be tested.

        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contains certain word embedding 
            pretrained model.

        return_effect_size : bool, optional
            Specifies if the returned score in 'result' field of results dict 
            is by default WEAT effect size metric, by default False
        
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
            and the scores of WEAT and the effect size of the metric.
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
                'query_name': query.query_name,
                'result': np.nan,
                'weat': np.nan,
                'effect_size': np.nan
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

        weat_effect_size = self._calc_effect_size(target_0, target_1, attribute_0,
                                                  attribute_1)
        weat = self._calc_weat(target_0, target_1, attribute_0, attribute_1)

        if calc_p_value:
            p_value = self._calc_p_value(
                target_embeddings,
                attribute_0,
                attribute_1,
                weat_original_result=weat,
                max_permutations=max_p_value_permutations,
                log_p_value_status=log_p_value_status,
            )
        else:
            p_value = None

        # return in result field effect_size
        if return_effect_size:
            return {
                'query_name': query.query_name,
                'result': weat_effect_size,
                'weat': weat,
                'effect_size': weat_effect_size,
                'p-value': p_value
            }

        # return in result field weat
        return {
            'query_name': query.query_name,
            'result': weat,
            'weat': weat,
            'effect_size': weat_effect_size,
            'p-value': p_value
        }
