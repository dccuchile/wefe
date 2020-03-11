import numpy as np
import logging

from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from ..utils import get_embeddings_from_word_set, verify_metric_input


class MAC():

    def __init__(self, disable_vocab_warnings: bool = True):
        self.template_required = (1, 'n')
        self.method_name = 'Mean Average Cosine Similarity (MAC)'
        self.abbreviated_method_name = 'MAC'
        self.disable_vocab_warnings = disable_vocab_warnings

    def calc_s(self, t, A_j):

        def calc_cos_dist(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return 1 / len(A_j) * np.sum([calc_cos_dist(t, a) for a in A_j])

    def calc_mac(self, T, A):
        first_term = 1 / (len(T))
        mac = first_term * np.sum([np.sum([self.calc_s(t_i, A_j) for A_j in A]) for t_i in T])

        return mac

    def run_query(self, query: Query, word_embeddings_wrapper: WordEmbeddingModel):

        verify_metric_input(query, word_embeddings_wrapper, self.template_required, self.method_name)

        T = get_embeddings_from_word_set(query.target_sets[0], word_embeddings_wrapper,
                                         warn_filtered_words=self.disable_vocab_warnings)

        A = [
            get_embeddings_from_word_set(current_attribute_set, word_embeddings_wrapper,
                                         warn_filtered_words=self.disable_vocab_warnings)
            for current_attribute_set in query.attribute_sets
        ]

        A_names = query.attribute_sets_names
        T_name = query.target_sets_names[0]
        return {'exp_name': "{} wrt {}".format(T_name, A_names), 'result': self.calc_mac(T, A)}
