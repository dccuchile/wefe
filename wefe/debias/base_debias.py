"""Contains a base class for implement any debias method in WEFE."""
from wefe.word_embedding_model import WordEmbeddingModel
from abc import abstractmethod


class BaseDebias:
    """Base class for implement any debias method in WEFE."""

    name = "BaseDebias Class"

    @abstractmethod
    def run_debias(self, word_embedding_model: WordEmbeddingModel, *args, **kwargs):
        """Execute a debias method over the provided word embedding model.

        Parameters
        ----------
        word_embedding_model : WordEmbeddingModel
            [description]
        """
        pass

    @abstractmethod
    def save(self):
        pass
