"""Contains a base class for implement any debias method in WEFE."""
from wefe.word_embedding_model import WordEmbeddingModel
from abc import abstractmethod


class BaseDebias:
    """Base class for implement any debias method in WEFE."""

    name = "BaseDebias Class"

    @abstractmethod
    def run_debias(
        self,
        word_embedding_model: WordEmbeddingModel,
        inplace: bool = True,
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> WordEmbeddingModel:
        """Execute a debias method over the provided word embedding model.

        Parameters
        ----------
        word_embedding_model : WordEmbeddingModel
            A word embedding model object.
        inplace : bool, optional
            Indicates whether the debiasing is performed inplace (i.e., the original
            embeddings are replaced by the new debiased ones) or a new model is created
            and the original embeddings are kept.

            **WARNING:** Inplace == False requires at least 2x RAM of the size of the
            model. Otherwise the execution of the debias probably will rise
            `MemoryError`.

        verbose : bool, optional
            Indicates whether the execution status of this function is printed in
            the logger, by default True.

        Returns
        -------
        WordEmbeddingModel
            A word embeddings model that has been debiased.

        """
        pass
