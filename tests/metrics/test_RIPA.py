"""RIPA metric testing."""
import numpy as np
from wefe.metrics import RIPA
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


def test_RIPA(model: WordEmbeddingModel, query_2t1a_1: Query):

    ripa = RIPA()

    results = ripa.run_query(query_2t1a_1, model)

    assert results["query_name"] == "Flowers and Insects wrt Pleasant"
    assert isinstance(results["result"], (np.number, float))
    assert isinstance(results["ripa"], (np.number, float))
    assert isinstance(results["word_values"], dict)

    for word, word_value in results["word_values"].items():
        assert isinstance(word, str)
        assert isinstance(word_value, dict)
        assert isinstance(word_value["mean"], (np.number, float))
        assert isinstance(word_value["std"], (np.number, float))
