"""A Word Embedding contanier based on gensim BaseKeyedVectors."""
from typing import Callable, Dict, Sequence, Union

import gensim
import numpy as np
import semantic_version

gensim_version = semantic_version.Version.coerce(gensim.__version__)
if gensim_version.major >= 4:
    from gensim.models import KeyedVectors as BaseKeyedVectors
else:
    from gensim.models.keyedvectors import BaseKeyedVectors


EmbeddingDict = Dict[str, np.ndarray]
EmbeddingSets = Dict[str, EmbeddingDict]


class WordEmbeddingModel:
    """A wrapper for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """

    def __init__(
        self, wv: BaseKeyedVectors, name: str = None, vocab_prefix: str = None
    ):
        """Initialize the word embedding model.

        Parameters
        ----------
        wv : BaseKeyedVectors.
            An instance of word embedding loaded through gensim KeyedVector
            interface or gensim's api.
        name : str, optional
            The name of the model, by default ''.
        vocab_prefix : str, optional.
            A prefix that will be concatenated with all word in the model
            vocab, by default None.

        Raises
        ------
        TypeError
            if word_embedding is not a KeyedVectors instance.
        TypeError
            if model_name is not None and not an instance of str.
        TypeError
            if vocab_prefix is not None and not an instance of str.


        Examples
        --------
        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from wefe.word_embedding_model import WordEmbeddingModel

        >>> dummy_model = Word2Vec(common_texts, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.name)
        Dummy model dim=10
        >>> print(model.vocab_prefix)
        /en/


        Attributes
        ----------
        wv : BaseKeyedVectors
            The model.
        vocab :
            The vocabulary of the model (a dict with the words that have an associated
            embedding in the model).
        model_name : str
            The name of the model.
        vocab_prefix : str
            A prefix that will be concatenated with each word of the vocab
            of the model.

        """
        # Type checking
        if not isinstance(wv, BaseKeyedVectors):
            raise TypeError(
                "wv should be an instance of gensim's BaseKeyedVectors"
                ", got {}.".format(wv)
            )

        if not isinstance(name, (str, type(None))):
            raise TypeError("name should be a string or None, got {}.".format(name))

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError(
                "vocab_prefix should be a string or None, got {}".format(vocab_prefix)
            )

        # Assign the attributes
        self.wv = wv

        # Obtain the vocabulary
        if gensim_version.major == 4:
            self.vocab = wv.key_to_index
        else:
            self.vocab = wv.vocab

        self.vocab_prefix = vocab_prefix
        if name is None:
            self.name = "Unnamed model"
        else:
            self.name = name

    def __eq__(self, other) -> bool:
        """Check if other is the same WordEmbeddingModel that self.

        Parameters
        ----------
        other : Any
            Some object

        Returns
        -------
        bool
            True if other is a WordEmbeddingModel that have the same model, model_name
            and vocab_prefix . False in any other case
        """
        if not isinstance(other, WordEmbeddingModel):
            return False
        if self.wv != other.wv:
            return False
        if self.name != other.name:
            return False
        if self.vocab_prefix != other.vocab_prefix:
            return False
        return True

    def __getitem__(self, key: str) -> Union[np.ndarray, None]:
        """Given a word, returns its associated embedding or none if it does not exist.

        Parameters
        ----------
        key : str
            A word

        Returns
        -------
        Union[np.ndarray, None]
            The embedding associated with the word or none if none if the word does not
            exist in the model.
        """
        if key in self.vocab:
            return self.wv[key]
        else:
            return None

    def __contains__(self, key):
        """Check if a word exists in the model's vocabulary.

        Parameters
        ----------
        key : str
            Some word.

        Returns
        -------
        bool
            True if the word exists in the model's vocabulary.
        """
        return key in self.vocab

    def normalize(self):
        """Normalize word embeddings in the model by using the L2 norm.

        Use the `init_sims` function of the gensim's `KeyedVectors` class.
        **Warning**: This operation is inplace. In other words, it replaces the
        embeddings with their L2 normalized versions.

        """
        if hasattr(self.wv, "init_sims"):
            self.wv.init_sims(replace=True)
        else:
            raise TypeError("The model does not have the init_sims method implemented.")

    def update(self, word: str, embedding: np.ndarray):
        """Update the value of an embedding of the model.

        If the method is executed with a word that is not in the vocabulary, an
        exception will be raised.

        Parameters
        ----------
        word : str
            The word whose embedding will be replaced. This word must be in the model's
            vocabulary.
        embedding : np.ndarray
            An embedding representing the word. It must have the same dimensions and
            data type as the model embeddings.

        Raises
        ------
        TypeError
            if word is not a1 string.
        TypeError
            if embedding is not an np.array.
        ValueError
            if word is not in the model's vocabulary.
        ValueError
            if the embedding is not the same size as the size of the model's embeddings.
        ValueError
            if the dtype of the embedding values is not the same as the model's
            embeddings.
        """
        if not isinstance(word, str):
            raise TypeError(
                f"word should be a string, got {word} with type {type(word)}."
            )

        if word not in self:
            raise ValueError(f"word '{word}' not in model vocab.")

        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                f"'{word}' new embedding should be a np.array,"
                f" got {type(embedding)}."
            )

        embedding_size = embedding.shape[0]
        if self.wv.vector_size != embedding_size:
            raise ValueError(
                f"The size of '{word}' embedding ({embedding_size}) is different from "
                f"the size of the embeddings in the model ({self.wv.vector_size})."
            )

        if not np.issubdtype(self.wv.vectors.dtype, embedding.dtype):
            raise ValueError(
                f"embedding dtype ({embedding.dtype}) is not the same of model's dtype "
                f"({self.wv.vectors.dtype})."
            )

        if gensim_version.major >= 4:
            word_index = self.wv.key_to_index[word]
        else:
            word_index = self.wv.vocab[word].index

        self.wv.vectors[word_index] = embedding

    def batch_update(
        self, words: Sequence[str], embeddings: Union[Sequence[np.ndarray], np.ndarray],
    ):
        """Update a batch of embeddings.

        This method calls `update_embedding` method with each of the word-embedding
        pairs.
        All words must be in the vocabulary, otherwise an exception will be thrown.
        Note that both `words` and `embeddings` must have the same number of elements,
        otherwise the method will raise an exception.

        Parameters
        ----------
        words : Sequence[str]
            A sequence (list, tuple or np.array) that contains the words whose
            representations will be updated.

        embeddings : Union[Sequence[np.ndarray], np.array],
            A sequence (list or tuple) or a np.array of embeddings or an np.array that
            contains all the new embeddings. The embeddings must have the same size and
            data type as the model.

        Raises
        ------
        TypeError
            if words is not a list
        TypeError
            if embeddings is not an np.ndarray
        Exception
            if words collection has not the same size of the embedding array.
        """
        if not isinstance(words, (list, tuple, np.ndarray)):
            raise TypeError(
                "words argument should be a list, tuple or np.array of strings, "
                f"got {type(words)}."
            )

        if not isinstance(embeddings, (list, tuple, np.ndarray)):
            raise TypeError(
                "embeddings should be a list, tuple or np.array, "
                f"got: {type(embeddings)}"
            )

        if len(words) != len(embeddings):
            raise ValueError(
                "words and embeddings must have the same size, got: "
                f"len(words) = {len(words)}, len(embeddings) = {len(embeddings)}"
            )

        for idx, word in enumerate(words):
            self.update(word, embeddings[idx])
