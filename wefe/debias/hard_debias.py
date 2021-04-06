"""Bolukbasi et al. Hard Debias WEFE implementation."""
import logging
from copy import deepcopy
from typing import Dict, List, Any, Sequence, Set, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA

from wefe.word_embedding_model import EmbeddingDict, WordEmbeddingModel
from wefe.debias.base_debias import BaseDebias

logger = logging.getLogger(__name__)


class HardDebias(BaseDebias):
    """Hard Debias binary debiasing method implementation on WEFE.

    This method allows to reduce the bias of an embedding model through geometric
    operations between embeddings.
    This method is binary because it only allows 2 classes of the same bias criterion,
    such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites), it is recommended
    to visit MulticlassHardDebias class.

    The main idea of this method is:

    1. to identify a bias subspace through the defining sets. In the case of gender,
    these could be e.g. `{'woman', 'man'}, {'she', 'he'}, ...`

    2. Neutralize the bias subspace of embeddings that should not be biased.
    First, it is defined a set of words that are correct to be related to the bias
    criterion: the *criterion specific gender words*.
    For example, in the case of gender, *gender specific* words are:
    `{'he', 'his', 'He', 'her', 'she', 'him', 'him', 'She', 'man', 'women', 'men'...}`.

    Then, it is defined that all words outside this set should have no relation to the
    bias criterion and thus have the possibility of being biased. (e.g. for the case of
    gender: `{doctor, nurse, ...}`). Therefore, this set of words is neutralized with
    respect to the bias subspace found in the previous step.

    The neutralization is carried out under the following operation:

    - u : embedding
    - v : bias direction

    First calculate the projection of the embedding on the bias subspace.
    - bias_subspace = v • (v • u) / (v • v)

    Then subtract the projection from the embedding.
    - u' = u - bias_subspace

    3. Equalization


    References
    ----------
    Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016).
    Man is to computer programmer as woman is to homemaker? debiasing word embeddings.
    Advances in Neural Information Processing Systems.

    https://github.com/tolga-b/debiaswe
    """

    name = "Bolukbasi's Hard Debias"
    short_name = "HD"

    def _identify_bias_subspace(
        self,
        definning_pairs_embeddings: List[EmbeddingDict],
        pca_args: Union[None, Dict[str, Any]] = None,
        n_components=10,
        verbose: bool = False,
    ) -> PCA:

        matrix = []
        for embedding_dict_pair in definning_pairs_embeddings:

            # Get the center of the current definning pair.
            pair_embeddings = np.array(list(embedding_dict_pair.values()))
            center = np.mean(pair_embeddings, axis=0)
            # For each word, embedding in the definning pair:
            for embedding in embedding_dict_pair.values():
                # Substract the center of the pair to the embedding
                matrix.append(embedding - center)
        matrix = np.array(matrix)  # type: ignore

        if pca_args is not None:
            pca = PCA(**pca_args)

        else:
            pca = PCA(n_components=n_components)

        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            if len(explained_variance) > 10:
                logger.info(f"PCA variance explaned: {explained_variance[0:10]}")
            else:
                logger.info(f"PCA variance explaned: {explained_variance}")

        return pca

    def _check_sets_size(
        self,
        sets: Sequence[Union[List[str], Tuple[str], Set[str], np.ndarray]],
        set_name: str,
    ):

        for idx, set_ in enumerate(sets):
            if len(set_) != 2:
                raise ValueError(
                    f"The {set_name} pair at position {idx} has less/more elements "
                    "than allowed by this method. "
                    f"Number of elements: {len(set_)} - Allowed: 2. "
                    f"Words in the set: {set_}"
                )

    def _drop(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Neutralize the bias of an embedding.

        Idea: First calculate the projection of the embedding on the bias subspace.
        - bias_subspace = v • (v • u) / (v • v)

        Then subtract the projection from the embedding.
        - u' = u - bias_subspace

        Note that this operation does not return the normalized embedding.

        Parameters
        ----------
        u : np.ndarray
            A embedding.
        v : np.ndarray
            A vector that represents the bias subspace.

        Returns
        -------
        np.ndarray
            A neutralized embedding.
        """
        return u - v * u.dot(v) / v.dot(v)

    def _neutralize_embeddings(
        self,
        embedding_model: WordEmbeddingModel,
        bias_direction: np.ndarray,
        bias_criterion_specific_words: Set[str],
    ):

        info_log = np.linspace(0, len(embedding_model.vocab), 11, dtype=np.int)

        for idx, word in enumerate(embedding_model.vocab):
            if word not in bias_criterion_specific_words:
                current_embedding = embedding_model[word]
                neutralized_embedding = self._drop(current_embedding, bias_direction)
                embedding_model.update_embedding(word, neutralized_embedding)

            if idx in info_log:
                logger.info(
                    f"Progress: {np.trunc(idx/ len(embedding_model.vocab) * 100)}% "
                    f"- Current word index: {idx}"
                )

    def _equalize_embeddings(
        self,
        embedding_model: WordEmbeddingModel,
        equalize_pairs_embeddings: List[EmbeddingDict],
        bias_direction: np.ndarray,
    ):
        for equalize_pair_embeddings in equalize_pairs_embeddings:
            (
                (word_a, embedding_a),
                (word_b, embedding_b,),
            ) = equalize_pair_embeddings.items()

            y = self._drop((embedding_a + embedding_b) / 2, bias_direction)

            z = np.sqrt(1 - np.linalg.norm(y) ** 2)

            if (embedding_a - embedding_b).dot(bias_direction) < 0:
                z = -z

            new_a = z * bias_direction + y
            new_b = -z * bias_direction + y

            # Update the embedding set with the equalized embeddings
            embedding_model.update_embedding(word_a, new_a)
            embedding_model.update_embedding(word_b, new_b)

    def run_debias(
        self,
        word_embedding_model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        bias_criterion_specific_words: Sequence[str],
        equalize_pairs: Sequence[Sequence[str]],
        pca_args: Dict[str, Any] = {"n_components": 10},
        debias_criterion_name: Union[str, None] = None,
        inplace: bool = True,
        verbose: bool = True,
    ) -> WordEmbeddingModel:
        """Execute Bolukbasi's Hard Debias.

        Parameters
        ----------
        word_embedding_model : WordEmbeddingModel
            A word embedding model to debias.

        definitional_pairs : Sequence[Sequence[str]],
            [description]

        bias_criterion_specific_words : Sequence[str],
            A collection of words that is specific to the bias criteria and must not be
            debiased (i.e., it is good that these words are associated with a certain
            social group).
            e.g.:`["spokesman", "wife", "himself", "son", "mother", "father",...]`

        equalize_pairs : Sequence[Sequence[str]],
            [description]

        pca_args : Dict[str, Any], optional
            Arguments for calculating the PCA, by default {"n_components": 10}.
            These arguments are the same as those of the `sklearn.decomposition.PCA`
            class.

        debias_criterion_name : Union[str, None], optional
            Name of the criterion on which the debias is being performed,
            This name will be included in the name of the returned model,
            by default None

        inplace : bool, optional
            Indicates whether the debiasing is performed inplace (i.e., the original
            embeddings are replaced by the new debiased ones) or a new model is created
            and the original embeddings are kept.

            **WARNING:** Inplace == False requires at least 2x RAM of the size of the
            model. Otherwise the execution of the debias probably will rise
            `MemoryError`.

            By default True

        verbose : bool, optional
            Indicates whether the execution status of this method is printed in
            the logger, by default True.

        Returns
        -------
        WordEmbeddingModel
            A word embeddings model that has been debiased.
        """
        logger.info(f"Executing Hard Debias on {word_embedding_model.model_name}")

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if not inplace:
            logger.warning(
                "Inplace argument is False. This method will attempt to create a copy "
                "of the original model. This may fail due to lack of memory."
            )
            word_embedding_model = deepcopy(word_embedding_model)
            logger.info("Copy created successfully.")
        else:
            logger.warning(
                "Inplace argument is True. The execution of this method will mutate "
                "the embeddings of the provided model."
            )

        self._check_sets_size(definitional_pairs, "definitional")
        self._check_sets_size(equalize_pairs, "equalize")

        # ------------------------------------------------------------------------------:
        # Normalize embeddings
        logger.debug("Normalizing embeddings.")
        word_embedding_model.normalize_embeddings()

        # ------------------------------------------------------------------------------:
        # Obtain the embedding of the definitional pairs.
        logger.debug("Obtaining definitional pairs.")
        self._definitional_pairs_embeddings = word_embedding_model.get_embeddings_from_sets(
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=True,
            verbose=verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        logger.debug("Identifying the bias subspace.")
        self._pca = self._identify_bias_subspace(
            self._definitional_pairs_embeddings, pca_args, verbose,
        )
        self._bias_direction = self._pca.components_[0]

        # ------------------------------------------------------------------------------
        # Neutralize the embeddings appointed in bias_criterion_specific_words:
        logger.debug("Neutralizing embeddings")
        self._neutralize_embeddings(
            word_embedding_model,
            self._bias_direction,
            set(bias_criterion_specific_words),
        )

        logger.debug("Normalizing embeddings.")
        word_embedding_model.normalize_embeddings()
        # ------------------------------------------------------------------------------
        # Equalize embeddings:

        # Get the equalization pairs candidates
        equalize_pairs_candidates = [
            x
            for e1, e2 in equalize_pairs
            for x in [
                (e1.lower(), e2.lower()),
                (e1.title(), e2.title()),
                (e1.upper(), e2.upper()),
            ]
        ]

        # Get the equalization pairs embeddings candidates
        logger.debug("Obtaining equalize pairs.")
        self._equalize_pairs_embeddings = word_embedding_model.get_embeddings_from_sets(
            sets=equalize_pairs_candidates,
            sets_name="equalize",
            warn_lost_sets=True,
            verbose=verbose,
        )

        # Execute the equalization
        logger.debug("Equalizing embeddings..")
        self._equalize_embeddings(
            word_embedding_model, self._equalize_pairs_embeddings, self._bias_direction,
        )

        # ------------------------------------------------------------------------------
        logger.debug("Normalizing embeddings.")
        word_embedding_model.normalize_embeddings()

        # ------------------------------------------------------------------------------
        # # Generate the new KeyedVectors
        if debias_criterion_name is not None:
            new_model_name = (
                f"{word_embedding_model.model_name}_{debias_criterion_name}_debiased"
            )
        else:
            new_model_name = (
                f"{word_embedding_model.model_name}_{debias_criterion_name}_debiased"
            )
        word_embedding_model.model_name = new_model_name

        logger.info("Done!")
        logger.setLevel(logging.INFO)

        return word_embedding_model

