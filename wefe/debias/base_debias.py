"""Contains a base class for implement any debias method in WEFE."""
from typing import List, Optional
from abc import abstractmethod

from sklearn.base import BaseEstimator
from wefe.word_embedding_model import WordEmbeddingModel


class BaseDebias(BaseEstimator):
    """Mixin class for implement any debias method in WEFE."""

    @abstractmethod
    def fit(
        self,
        model: WordEmbeddingModel,
        debias_criterion_name: Optional[str] = None,
        **fit_params,
    ) -> "BaseDebias":
        """Fit the transformation.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        debias_criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g. 'Gender'. This will indicate the name of the model returning transform,
            by default None
        verbose: bool, optional
            True will print informative messages about the debiasing process,
            by default False.

        """
        raise NotImplementedError()

    @abstractmethod
    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Perform the debiasing method over the model provided.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be performed
            only on the word embeddings of this set. In the case of provide `None`, the
            debias will be performed on all words (except those specified in ignore).
            by default `None`.
        ignore : Optional[List[str]], optional
            If target is `None` and a set of words is specified in ignore, the debias
            method will perform the debias in all words except those specified in this
            set, by default `None`.
        copy : bool, optional
            If `True`, the debias will be performed on a copy of the model.
            If `False`, the debias will be applied on the same model delivered, causing
            its vectors to mutate.
            **WARNING:** Setting copy with `True` requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may raise
            `MemoryError`, by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased word embedding model.
        """
        raise NotImplementedError()

    def fit_transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
        **fit_params,
    ) -> WordEmbeddingModel:
        """Convenience method to execute fit and transform in a single call.

        Parameters
        ----------
        model : WordEmbeddingModel
            A word embedding model object.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be applied
            only on the word embeddings of this set, by default None.
        ignore : Optional[List[str]], optional
            If target is None and a set of words is specified in ignore, the debias
            method will debias all words except those specified in ignore,
            by default None.
        copy : bool, optional
            If True, the debias will be performed on a copy of the model.
            If False, the debias will be applied on the same model delivered, causing
            its vectors to mutate.
            **WARNING:** Setting copy with True requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may raise
            `MemoryError`, by default True.
        verbose : bool, optional
            True will print informative messages about the debiasing process,
            by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased word embedding model.
        """
        return self.fit(model, **fit_params).transform(
            model, target=target, ignore=ignore, copy=copy
        )

    def _check_transform_args(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
    ):
        # check model
        if not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                f"model should be a WordEmbeddingModel instance, got {model}."
            )

        # check target
        if target is not None and not isinstance(target, list):
            raise TypeError(
                f"target should be None or a list of strings, got {target}."
            )

        if isinstance(target, list):
            for idx, word in enumerate(target):
                if not isinstance(word, str):
                    raise TypeError(
                        "All elements in target should be strings"
                        f", got: {word} at index {idx} "
                    )

        # check ignore
        if ignore is not None and not isinstance(ignore, list):
            raise TypeError(
                f"ignore should be None or a list of strings, got {ignore}."
            )

        if isinstance(ignore, list):
            for idx, word in enumerate(ignore):
                if not isinstance(word, str):
                    raise TypeError(
                        "All elements in ignore should be strings"
                        f", got: {word} at index {idx} "
                    )

        # check copy
        if not isinstance(copy, bool):
            raise TypeError(f"copy should be a bool, got {copy}.")
