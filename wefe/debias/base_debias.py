"""Contains a base class for implement any debias method in WEFE."""

from abc import abstractmethod
from typing import Optional, Union

from wefe.word_embedding_model import WordEmbeddingModel


class BaseDebias:
    """Mixin class for implement any debias method in WEFE."""

    # The name of the method.
    name: str

    @abstractmethod
    def fit(
        self,
        model: WordEmbeddingModel,
        **fit_params,
    ) -> "BaseDebias":
        """Fit the transformation.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        verbose: bool, optional
            True will print informative messages about the debiasing process,
            by default False.

        """
        raise NotImplementedError()

    @abstractmethod
    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Perform the debiasing method over the model provided.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be
            performed only on the word embeddings of this set. In the case of provide
            `None`, the debias will be performed on all words (except those specified
            in ignore).
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
        target: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
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
        target: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
        copy: bool = True,
    ) -> None:
        # check if model is a WordEmbeddingModel
        if not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                f"model should be a WordEmbeddingModel instance, got {model}."
            )

        # check if target is a list
        if target is not None and not isinstance(target, list):
            raise TypeError(
                f"target should be None or a list of strings, got {target}."
            )

        # check if all elements of the list are strings
        if isinstance(target, list):
            for idx, word in enumerate(target):
                if not isinstance(word, str):
                    raise TypeError(
                        "All elements in target should be strings"
                        f", got: {word} at index {idx} "
                    )

        # check ignore parameter type (none or list)
        if ignore is not None and not isinstance(ignore, list):
            raise TypeError(
                f"ignore should be None or a list of strings, got {ignore}."
            )

        # if ignore is a list, check that all its elements are string
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

    def _check_sets_sizes(
        self,
        sets: list[list[str]],
        set_name: str,
        set_size: Union[int, str],
    ) -> None:
        if len(sets) == 0:
            raise ValueError("")

        # case fixed set_size.
        if isinstance(set_size, int):
            for idx, set_ in enumerate(sets):
                if len(set_) != set_size:
                    adverb = "less" if len(set_) < set_size else "more"

                    raise ValueError(
                        f"The {set_name} set at position {idx} ({set_}) has {adverb} "
                        f"words than allowed by {self.name}: "
                        f"got {len(set_)} words, expected {set_size}."
                    )

        # case free set_size.
        elif set_size == "n":
            inferred_set_size = len(sets[0])

            for idx, set_ in enumerate(sets):
                if len(set_) != inferred_set_size:
                    adverb = "less" if len(set_) < inferred_set_size else "more"

                    raise ValueError(
                        f"The {set_name} set at position {idx} ({set_}) has {adverb} "
                        f"words than the other {set_name} sets: "
                        f"got {len(set_)} words, expected {inferred_set_size}."
                    )

        else:
            raise ValueError('Wrong set_size value {set_size}. Expected int or "n"')
