from gensim.models import KeyedVectors


class WordEmbeddingModel:
    """A wrapper for Word Embedding pre-trained model using gensim's KeyedVectors. It includes the name of the model and some vocab prefix if needed.
    """

    def __init__(self, word_embedding: KeyedVectors, model_name: str = None, vocab_prefix: str = None):
        """Initializes the WordEmbeddingModel Wrapper. 
        
        Parameters
        ----------
        word_embedding : KeyedVectors
            An instance of word embedding loaded through gensim KeyedVector interface.
        model_name : str, optional
            The name of the model, by default ''.
        vocab_prefix : str, optional
            A prefix that will be concatenated with all word in the model vocab, by default None.
        
        Raises
        ------
        TypeError
            if word_embedding is not a KeyedVectors instance
        TypeError
            if model_name is not None and not instance of str
        TypeError
            if vocab_prefix is not None and not instance of str
        """

        if not isinstance(word_embedding, KeyedVectors):
            raise TypeError(
                "word_embedding must be an instance of a gensim's KeyedVectors. Given: {}".format(word_embedding))
        else:
            self.word_embedding = word_embedding

        if model_name is None:
            self.model_name = 'Unnamed word embedding model'
        elif not isinstance(model_name, str):
            raise TypeError('model_name must be a string. Given: {}'.format(model_name))
        else:
            self.model_name = model_name

        if vocab_prefix is None:
            self.vocab_prefix = ''
        elif not isinstance(vocab_prefix, str):
            raise TypeError('vocab_prefix parameter must be a string. Given: {}'.format(vocab_prefix))
        else:
            self.vocab_prefix = vocab_prefix

    def __eq__(self, other):
        if self.word_embedding != other.word_embedding:
            return False
        if self.model_name != other.model_name:
            return False
        if self.model_name != other.model_name:
            return False
        return True
