"""Contains a base class for implement any debias method in WEFE."""
from typing import List, Union
from abc import abstractmethod

from sklearn.base import BaseEstimator
from wefe.word_embedding_model import WordEmbeddingModel


class BaseDebias(BaseEstimator):
    """Mixin class for implement any debias method in WEFE."""

    @abstractmethod
    def fit(
        self, model: WordEmbeddingModel, verbose: bool = True, **fit_params
    ) -> "BaseDebias":
        """Compute the transformation that will be applied later.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        verbose: bool, optional
            If true, it prints information about the debias status at each step,
            by default True.

        """
        return self

    @abstractmethod
    def transform(
        self,
        model: WordEmbeddingModel,
        ignore: Union[List[str], None] = None,
        copy: bool = True,
        verbose: bool = True,
        **transform_params,
    ) -> WordEmbeddingModel:
        """Perform the debiasing method over the model provided.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        copy : bool, optional
            If True, perform the debiasing on a copy of the model.
            If False, apply the debias on the model provided.
            **WARNING:** Setting copy with True requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may rise
            `MemoryError`, by default True.
        ignore : List[str], optional
            A list of words that will be ignored in the debiasing process.
            Check the compatibility of this parameter with each method.
            by default None.
        verbose: bool, optional
            If true, it prints information about the transform status at each step,
             by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.
        """
        pass

    def fit_transform(
        self,
        model: WordEmbeddingModel,
        ignore: Union[List[str], None] = None,
        copy: bool = True,
        verbose: bool = True,  # TODO: Cambiar esto por False para el deploy
        **fit_params,
    ):
        """Convenience method to execute fit and transform in a single call.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        ignore : Union[List[str], None], optional
           A list of words that will be ignored in the debiasing process,
           by default None.
        copy : bool, optional
            If True, perform the debiasing on a copy of the model.
            If False, apply the debias on the model provided.
            **WARNING:** Setting copy with True requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may rise
            `MemoryError`. by default True.
        verbose : bool, optional
            [description], by default True

        Returns
        -------
        [type]
            [description]
        """

        return self.fit(model, verbose=verbose, **fit_params).transform(
            model, ignore=ignore, copy=copy, verbose=verbose
        )

    # @abstractmethod
    # def run_debias(
    #     self,
    #     word_embedding_model: WordEmbeddingModel,
    #     inplace: bool = True,
    #     verbose: bool = True,
    #     *args,
    #     **kwargs,
    # ) -> WordEmbeddingModel:
    #     """Execute a debias method over the provided word embedding model.

    #     Parameters
    #     ----------
    #     word_embedding_model : WordEmbeddingModel
    #         A word embedding model object.
    #     inplace : bool, optional
    #         Indicates whether the debiasing is performed inplace (i.e., the original
    #         embeddings are replaced by the new debiased ones) or a new model is created
    #         and the original embeddings are kept.

    #     verbose : bool, optional
    #         Indicates whether the execution status of this function is printed in
    #         the logger, by default True.

    #     Returns
    #     -------
    #     WordEmbeddingModel
    #         A word embeddings model that has been debiased.

    #     """
    #     pass
