from gensim.models.keyedvectors import BaseKeyedVectors


class WordEmbeddingModel:
    """A container for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """
    def __init__(self,
                 word_embedding: BaseKeyedVectors,
                 model_name: str = None,
                 vocab_prefix: str = None):
        """Initializes the WordEmbeddingModel container.

        Parameters
        ----------
        keyed_vectors : BaseKeyedVectors.
            An instance of word embedding loaded through gensim KeyedVector
            interface or gensim's api.
        model_name : str, optional
            The name of the model, by default ''.
        vocab_prefix : str, optional.
            A prefix that will be concatenated with all word in the model
            vocab, by default None.

        Raises
        ------
        TypeError
            if word_embedding is not a KeyedVectors instance.
        TypeError
            if model_name is not None and not instance of str.
        TypeError
            if vocab_prefix is not None and not instance of str.

        Examples
        --------
        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from wefe.word_embedding_model import WordEmbeddingModel

        >>> dummy_model = Word2Vec(common_texts, size=10, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.model_name_)
        Dummy model dim=10
        >>> print(model.vocab_prefix_)
        /en/


        Attributes
        ----------
        model_ : KeyedVectors
            The object that contains the model.
        model_name_ : str
            The name of the model.
        vocab_prefix_ : str
            A prefix that will be concatenated with each word of the vocab
            of the model.

        """

        if not isinstance(word_embedding, BaseKeyedVectors):
            raise TypeError('word_embedding must be an instance of a gensim\'s'
                            ' KeyedVectors. Given: {}'.format(word_embedding))
        else:
            self.model_ = word_embedding

        if model_name is None:
            self.model_name_ = 'Unnamed word embedding model'
        elif not isinstance(model_name, str):
            raise TypeError(
                'model_name must be a string. Given: {}'.format(model_name))
        else:
            self.model_name_ = model_name

        if vocab_prefix is None:
            self.vocab_prefix_ = ''
        elif not isinstance(vocab_prefix, str):
            raise TypeError(
                'vocab_prefix parameter must be a string. Given: {}'.format(
                    vocab_prefix))
        else:
            self.vocab_prefix_ = vocab_prefix

    def __eq__(self, other):
        """
        Return true if other_name

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        if self.model_ != other.model_:
            return False
        if self.model_name_ != other.model_name_:
            return False
        return True
