import pytest
from ..word_embedding_model import WordEmbeddingModel
from ..utils import load_weat_w2v


def test_create_word_embedding_model():

    # target sets None
    with pytest.raises(TypeError, match='word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel(None)

    # target sets int
    with pytest.raises(TypeError, match='word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel('abc')

    with pytest.raises(TypeError, match='word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel(1)

    with pytest.raises(TypeError, match='word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel({})

    with pytest.raises(TypeError, match='word_embedding must be an instance of a gensim\'s KeyedVectors'):
        WordEmbeddingModel({})

    weat_we = load_weat_w2v()

    with pytest.raises(TypeError, match='model_name must be a string'):
        WordEmbeddingModel(weat_we, 12)

    with pytest.raises(TypeError, match='vocab_prefix parameter must be a string.'):
        WordEmbeddingModel(weat_we, 'A', 12)

    model_1 = WordEmbeddingModel(weat_we, 'weat_we')
    assert model_1.model_ == weat_we
    assert model_1.model_name_ == 'weat_we'
    assert model_1.vocab_prefix_ == ''

    model_2 = WordEmbeddingModel(weat_we)
    assert model_2.model_name_ == 'Unnamed word embedding model'
    assert model_2.vocab_prefix_ == ''

    model_3 = WordEmbeddingModel(weat_we, 'weat_we', '\\c\\en')
    assert model_3.model_name_ == 'weat_we'
    assert model_3.vocab_prefix_ == '\\c\\en'
