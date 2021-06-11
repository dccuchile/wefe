"""A Word Embedding contanier based on gensim BaseKeyedVectors."""
from typing import Callable, Dict, Sequence, Union

import numpy as np
import gensim
import semantic_version

gensim_version = semantic_version.Version.coerce(gensim.__version__)
if gensim_version.major >= 4:
    from gensim.models import KeyedVectors as BaseKeyedVectors
else:
    from gensim.models.keyedvectors import BaseKeyedVectors

PreprocessorArgs = Dict[str, Union[bool, str, Callable, None]]

EmbeddingDict = Dict[str, np.ndarray]
EmbeddingSets = Dict[str, EmbeddingDict]


class WordEmbeddingModel:
    """A wrapper for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """

    def __init__(
        self, model: BaseKeyedVectors, model_name: str = None, vocab_prefix: str = None
    ):
        """Initialize the word embedding model.

        Parameters
        ----------
        model : BaseKeyedVectors.
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

        >>> dummy_model = Word2Vec(common_texts, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.model_name)
        Dummy model dim=10
        >>> print(model.vocab_prefix)
        /en/


        Attributes
        ----------
        model : BaseKeyedVectors
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
        if not isinstance(model, BaseKeyedVectors):
            raise TypeError(
                "model should be an instance of gensim's BaseKeyedVectors"
                ", got {}.".format(model)
            )

        if not isinstance(model_name, (str, type(None))):
            raise TypeError(
                "model_name should be a string or None, got {}.".format(model_name)
            )

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError(
                "vocab_prefix should be a string or None, got {}".format(vocab_prefix)
            )

        # Assign the attributes
        self.model = model

        # Obtain the vocabulary
        if gensim_version.major == 4:
            self.vocab = model.key_to_index
        else:
            self.vocab = model.vocab

        self.vocab_prefix = vocab_prefix
        if model_name is None:
            self.model_name = "Unnamed word embedding model"
        else:
            self.model_name = model_name

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
        if self.model != other.model:
            return False
        if self.model_name != other.model_name:
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
            return self.model[key]
        else:
            return None

    def normalize_embeddings(self):
        """Normalize word embeddings in the model by using the L2 norm.

        Use the `init_sims` function of the gensim's `KeyedVectors` class.
        **Warning**: This operation is inplace. In other words, it replaces the
        embeddings with their L2 normalized versions.

        Parameters
        ----------
        replace : bool, optional
            [description], by default True
        """
        if hasattr(self.model, "init_sims"):
            self.model.init_sims(replace=True)
        else:
            raise TypeError("The model does not have the init_sims method implemented.")

    def update_embedding(self, word: str, embedding: np.ndarray):
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
            if word is not a string.
        TypeError
            if embedding is not a np.array.
        ValueError
            if word is not in the model's vocabulary.
        ValueError
            if the embedding is not the same size as the size of the model's embeddings.
        ValueError
            if the dtype of the embedding values is not the same of the model's
            embeddings.
        """
        if not isinstance(word, str):
            raise TypeError(
                f"word should be a string, got {word} with type {type(word)}."
            )

        if word not in self.model.key_to_index:
            raise ValueError(f"word '{word}' not in model vocab.")

        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                f"'{word}' new embedding should be a np.array,"
                f" got {type(embedding)}."
            )

        embedding_size = embedding.shape[0]
        if self.model.vector_size != embedding_size:
            raise ValueError(
                f"The size of '{word}' embedding ({embedding_size}) is different from "
                f"the size of the embeddings in the model ({self.model.vector_size})."
            )

        if not np.issubdtype(self.model.vectors.dtype, embedding.dtype):
            raise ValueError(
                f"embedding dtype ({embedding.dtype}) is not the same of model's dtype "
                f"({self.model.vectors.dtype})."
            )

        if gensim_version.major >= 4:
            word_index = self.model.key_to_index[word]
        else:
            word_index = self.model.vocab[word].index

        self.model.vectors[word_index] = embedding

    def update_embeddings(
        self, words: Sequence[str], embeddings: Union[Sequence[np.ndarray], np.ndarray],
    ):
        """Update a list of embeddings.

        This method calls `update_embedding` method with each of the word-embedding
        pairs.
        All words must be in the vocabulary, otherwise an exception will be thrown.
        Note that both `words` and `embeddings`must have the same number of elements,
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
            if embeddings is not a np.ndarray
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
            self.update_embedding(word, embeddings[idx])
