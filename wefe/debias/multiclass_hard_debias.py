"""Manzini et al. Multiclass Hard Debias WEFE implementation."""
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from wefe.debias.base_debias import BaseDebias
from wefe.preprocessing import get_embeddings_from_sets
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import EmbeddingDict, WordEmbeddingModel

logger = logging.getLogger(__name__)


class MulticlassHardDebias(BaseDebias):
    """Generalized version of Hard Debias that enables multiclass debiasing.

    Generalized refers to the fact that this method extends Hard Debias in order to
    support more than two types of social target sets within the definitional set.
    For example, for the case of religion bias, it supports a debias using words
    associated with Christianity, Islam and Judaism.

    References
    ----------
    | [1]: Manzini, T., Chong, L. Y., Black, A. W., & Tsvetkov, Y. (2019, June).
    | Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass
    | Bias in Word Embeddings.
    | In Proceedings of the 2019 Conference of the North American Chapter of the
    | Association for Computational Linguistics: Human Language Technologies,
    | Volume 1 (Long and Short Papers) (pp. 615-621).
    | [2]: https://github.com/TManzini/DebiasMulticlassWordEmbedding
    """

    def __init__(
        self,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = False,
        criterion_name: Optional[str] = None,
    ) -> None:
        """Initialize a Multiclass Hard Debias instance.

        Parameters
        ----------
        pca_args : Dict[str, Any], optional
            Arguments for the PCA that is calculated internally in the identification
            of the bias subspace, by default {"n_components": 10}
        verbose : bool, optional
            True will print informative messages about the debiasing process,
            by default False.
        criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g. 'Gender'. This will indicate the name of the model returning transform,
            by default None
        """
        # check pca args
        if not isinstance(pca_args, dict):
            raise TypeError(f"pca_args should be a dict, got {pca_args}.")
        # check verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose should be a bool, got {verbose}.")

        self.pca_args = pca_args
        self.verbose = verbose

        if "n_components" in pca_args:
            self.pca_num_components_ = pca_args["n_components"]
        else:
            self.pca_num_components_ = 10

        if criterion_name is None or isinstance(criterion_name, str):
            self.criterion_name_ = criterion_name
        else:
            raise ValueError(
                f"debias_criterion_name should be str, got: {criterion_name}"
            )

    def _identify_bias_subspace(
        self, definning_sets_embeddings: List[EmbeddingDict],
    ) -> PCA:

        matrix = []
        for definning_set_dict in definning_sets_embeddings:

            # Get the center of the current definning pair.
            set_embeddings = np.array(list(definning_set_dict.values()))
            center = np.mean(set_embeddings, axis=0)
            # For each word, embedding in the definning pair:
            for embedding in definning_set_dict.values():
                # Substract the center of the pair to the embedding
                matrix.append(embedding - center)
        matrix = np.array(matrix)  # type: ignore

        pca = PCA(**self.pca_args)
        pca.fit(matrix)

        if self.verbose:
            explained_variance = pca.explained_variance_ratio_
            if len(explained_variance) > 10:
                logger.info(f"PCA variance explaned: {explained_variance[0:10]}")
            else:
                logger.info(f"PCA variance explaned: {explained_variance}")

        return pca

    def _project_onto_subspace(self, vector, subspace):
        v_b = np.zeros_like(vector)
        for component in subspace:
            v_b += np.dot(vector.transpose(), component) * component
        return v_b

    def _get_target(
        self, model: WordEmbeddingModel, target: Optional[Sequence[str]] = None,
    ) -> List[str]:

        definitional_words = np.array(self.definitional_sets_).flatten().tolist()

        if target is not None:
            # keep only words in the model's vocab.
            target = list(
                filter(
                    lambda x: x in model.vocab and x not in definitional_words, target,
                )
            )
        else:
            # indicate that all words are canditates to neutralize.
            target = list(
                filter(lambda x: x not in definitional_words, model.vocab.keys(),)
            )

        return target

    def _neutralize(
        self,
        model: WordEmbeddingModel,
        bias_subspace: np.ndarray,
        target: Optional[List[str]],
        ignore: Optional[List[str]],
    ):
        if target is not None:
            target_ = set(target)
        else:
            target_ = set(model.vocab.keys())

        if ignore is not None and target is None:
            ignore_ = set(ignore)
        else:
            ignore_ = set()

        for word in tqdm(target_):
            if word not in ignore_:
                # get the embedding
                v = model[word]
                # neutralize the embedding if the word is not in the definitional words.
                v_b = self._project_onto_subspace(v, bias_subspace)
                # neutralize the embedding
                new_v = (v - v_b) / np.linalg.norm(v - v_b)
                # update the old values
                model.update(word, new_v)

    def _equalize(
        self,
        model: WordEmbeddingModel,
        equalize_sets_embeddings: List[EmbeddingDict],
        bias_subspace: np.ndarray,
    ):
        for equalize_pair_embeddings in equalize_sets_embeddings:

            words = equalize_pair_embeddings.keys()
            embeddings = np.array(list(equalize_pair_embeddings.values()))

            # calculate the mean of the equality set
            mean = np.mean(embeddings, axis=0)
            # project the mean in the bias subspace
            mean_b = self._project_onto_subspace(mean, bias_subspace)
            # discard the projection from the mean
            upsilon = mean - mean_b

            for (word, embedding) in zip(words, embeddings):
                v_b = self._project_onto_subspace(embedding, bias_subspace)
                frac = (v_b - mean_b) / np.linalg.norm(v_b - mean_b)
                new_v = upsilon + np.sqrt(1 - np.sum(np.square(upsilon))) * frac
                model.update(word, new_v)

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_sets: Sequence[Sequence[str]],
        equalize_sets: Sequence[Sequence[str]],
    ) -> BaseDebias:
        """Compute the bias direction and obtains the equalize embedding pairs.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        definitional_sets : Sequence[Sequence[str]]
            A sequence of string pairs that will be used to define the bias direction.
            For example, for the case of gender debias, this list could be [['woman',
            'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ...].
        equalize_pairs : Optional[Sequence[Sequence[str]]], optional
            A list with pairs of strings, which will be equalized.
            In the case of passing None, the equalization will be done over the word
            pairs passed in definitional_sets,
            by default None.

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """
        # ------------------------------------------------------------------------------:
        # Obtain the embedding of the definitional sets.

        if self.verbose:
            print("Obtaining definitional sets.")
        self.definitional_sets_ = definitional_sets
        self.definitional_sets_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=definitional_sets,
            sets_name="definitional",
            warn_lost_sets=True,
            normalize=True,
            verbose=self.verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning sets.
        if self.verbose:
            print("Identifying the bias subspace.")
        self.pca_ = self._identify_bias_subspace(self.definitional_sets_embeddings_,)
        self.bias_subspace_ = self.pca_.components_[: self.pca_num_components_]

        # ------------------------------------------------------------------------------
        # Equalize embeddings:

        # Get the equalization sets embeddings.
        # Note that the equalization sets are the same as the definitional sets.
        if self.verbose:
            print("Obtaining equalize pairs.")
        self.equalize_sets_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=equalize_sets,
            sets_name="equalize",
            normalize=True,
            warn_lost_sets=True,
            verbose=self.verbose,
        )
        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Execute Multiclass Hard Debias over the provided model.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will be performed
            only on the word embeddings of this set. If `None` is provided, the
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
            **WARNING:** Setting copy with `True` requires RAM at least 2x of the size
            of the model, otherwise the execution of the debias may raise to
            `MemoryError`, by default True.
            
        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.
        """
        self._check_transform_args(
            model=model, target=target, ignore=ignore, copy=copy,
        )

        check_is_fitted(
            self,
            [
                "definitional_sets_",
                "definitional_sets_embeddings_",
                "pca_",
                "bias_subspace_",
            ],
        )

        if self.verbose:
            print(f"Executing Multiclass Hard Debias on {model.name}")

        # ------------------------------------------------------------------------------
        # Copy
        if copy:
            print(
                "copy argument is True. Transform will attempt to create a copy "
                "of the original model. This may fail due to lack of memory."
            )
            model = deepcopy(model)
            print("Model copy created successfully.")

        else:
            print(
                "copy argument is False. The execution of this method will mutate "
                "the original model."
            )

        # ------------------------------------------------------------------------------
        # Neutralize the embeddings provided in target.
        # if target is None, the debias will be performed in all embeddings.
        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()

        # Neutralize the embeddings:
        if self.verbose:
            print("Neutralizing embeddings")

        # get the words that will be debiased.
        target = self._get_target(model, target)
        self._neutralize(
            model=model,
            bias_subspace=self.bias_subspace_,
            target=target,
            ignore=ignore,
        )

        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()

        # ------------------------------------------------------------------------------
        # Equalize embeddings:

        # Execute the equalization
        logger.debug("Equalizing embeddings..")
        self._equalize(
            model=model,
            equalize_sets_embeddings=self.equalize_sets_embeddings_,
            bias_subspace=self.bias_subspace_,
        )

        # ------------------------------------------------------------------------------
        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()

        # ------------------------------------------------------------------------------
        # # Generate the new KeyedVectors
        if self.criterion_name_ is None:
            new_model_name = f"{model.name}_debiased"
        else:
            new_model_name = f"{model.name}_{self.criterion_name_}_debiased"
        model.name = new_model_name

        if self.verbose:
            print("Done!")

        return model
