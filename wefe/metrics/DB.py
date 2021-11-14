from typing import Union, Callable, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from wefe.metrics.base_metric import BaseMetric
from wefe.query import Query
from wefe.models.base_model import BaseModel


class DirectBias(BaseMetric):
    def run_query(
        self,
        query: Query,
        word_embedding: BaseModel,
        lost_vocabulary_threshold: float,
        preprocessor_options: Dict[str, Union[bool, str, Callable, None]],
        secondary_preprocessor_options: Union[
            Dict[str, Union[bool, str, Callable, None]], None
        ],
        warn_not_found_words: bool,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return super().run_query(
            query,
            word_embedding,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessor_options=preprocessor_options,
            secondary_preprocessor_options=secondary_preprocessor_options,
            warn_not_found_words=warn_not_found_words,
            *args,
            **kwargs
        )

    def calculateDirectBias(vocab, neutral_words, bias_subspace, c=1):
        directBiasMeasure = 0
        for word in neutral_words:
            vec = vocab[word]
            directBiasMeasure += (
                np.linalg.norm(cosine_similarity(vec, bias_subspace)) ** c
            )
        directBiasMeasure *= 1.0 / len(neutral_words)
        return directBiasMeasure
