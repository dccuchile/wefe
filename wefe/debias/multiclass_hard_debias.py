"""Manzini et al. Multiclass Hard Debias WEFE implementation."""
from functools import reduce
import logging
from copy import deepcopy
from math import sqrt
from typing import List, Optional, Sequence, Union, Dict, Any

import numpy as np
from sklearn.decomposition import PCA

from wefe.debias.hard_debias import HardDebias
from wefe.word_embedding_model import WordEmbeddingModel, EmbeddingDict

logger = logging.getLogger(__name__)


class MulticlassHardDebias(HardDebias):
    """Generalized version of Hard Debias that enables multiclass debiasing.

    Reference
    ---------
    Manzini, T., Chong, L. Y., Black, A. W., & Tsvetkov, Y. (2019, June).
    Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass
    Bias in Word Embeddings.
    In Proceedings of the 2019 Conference of the North American Chapter of the
    Association for Computational Linguistics: Human Language Technologies,
    Volume 1 (Long and Short Papers) (pp. 615-621).

    https://github.com/TManzini/DebiasMulticlassWordEmbedding

    """

    def _identify_bias_subspace(
        self,
        definning_sets_embeddings: List[EmbeddingDict],
        n_components: int,
        verbose: bool = False,
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

        pca = PCA(n_components=n_components)

        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            if len(explained_variance) > 10:
                logger.info(f"PCA variance explaned: {explained_variance[0:10]}")
            else:
                logger.info(f"PCA variance explaned: {explained_variance}")

        return pca

    def project_onto_subspace(self, vector, subspace):
        v_b = np.zeros_like(vector)
        for component in subspace:
            v_b += np.dot(vector.transpose(), component) * component
        return v_b

    def _neutralize_embeddings(
        self,
        word_embedding_model: WordEmbeddingModel,
        bias_subspace: np.ndarray,
        words_to_neutralize: Sequence[str],
    ):

        info_log = np.linspace(0, len(word_embedding_model.vocab), 11, dtype=np.int)

        for idx, word in enumerate(words_to_neutralize):
            embedding = word_embedding_model[word]
            # neutralize the embedding if the word is not in the definitional words.
            projection = self.project_onto_subspace(embedding, bias_subspace)
            neutralized_embedding = (embedding - projection) / np.linalg.norm(
                embedding - projection
            )
            word_embedding_model.update_embedding(word, neutralized_embedding)

            if idx in info_log:
                logger.info(
                    f"Progress: {np.trunc(idx/ len(word_embedding_model.vocab) * 100)}% "
                    f"- Current word index: {idx}"
                )

    def _get_words_to_neutralize(
        self,
        word_embedding_model: WordEmbeddingModel,
        definitional_sets: Sequence[Sequence[str]],
        words_to_neutralize: Optional[Sequence[str]] = None,
    ):

        definitional_words = np.array(definitional_sets).flatten().tolist()

        if words_to_neutralize is not None:
            # keep only words in the model's vocab.
            words_to_neutralize = list(
                filter(
                    lambda x: x in word_embedding_model.vocab
                    and x not in definitional_words,
                    words_to_neutralize,
                )
            )
        else:
            # indicate that neutralize all words.
            words_to_neutralize = list(
                filter(
                    lambda x: x not in definitional_words,
                    word_embedding_model.vocab.keys(),
                )
            )

        return words_to_neutralize

    def _equalize_embeddings(
        self,
        embedding_model: WordEmbeddingModel,
        equalize_sets_embeddings: List[EmbeddingDict],
        bias_subspace: np.ndarray,
    ):
        for equalize_pair_embeddings in equalize_sets_embeddings:

            words = equalize_pair_embeddings.keys()
            embeddings = equalize_pair_embeddings.values()

            mean = np.mean(np.array(embeddings), axis=1)
            mean_b = self.project_onto_subspace(mean, bias_subspace)
            upsilon = mean - mean_b

            for (word, embedding) in zip(words, embeddings):
                v_b = self.project_onto_subspace(embedding, bias_subspace)
                frac = (v_b - mean_b) / np.linalg.norm(v_b - mean_b)
                new_v = upsilon + np.sqrt(1 - np.sum(np.square(upsilon))) * frac
                embedding_model.update_embedding(word, new_v)

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_sets: Sequence[Sequence[str]],
        equalize_pairs: Optional[Sequence[Sequence[str]]] = None,
        num_pca_components: int = 10,
        debias_criterion_name: Optional[str] = None,
        verbose: bool = False,
    ) -> "BaseDebias":
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
            A list with pairs of strings which will be equalized.
            In the case of passing None, the equalization will be done over the word
            pairs passed in definitional_pairs,
            by default None.
        debias_criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g. 'Gender'. This will indicate the name of the model returning transform,
            by default None
        verbose : bool, optional
            True will print informative messages about the debiasing process,
            by default False.

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """

        # ------------------------------------------------------------------------------:
        # Obtain the embedding of the definitional sets.
        logger.debug("Obtaining definitional sets.")
        self._definitional_sets_embeddings = model.get_embeddings_from_sets(
            sets=definitional_sets,
            sets_name="definitional",
            warn_lost_sets=True,
            normalize=True,
            verbose=verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning sets.
        logger.debug("Identifying the bias subspace.")
        self._pca = self._identify_bias_subspace(
            self._definitional_sets_embeddings, verbose,
        )
        self._bias_direction = self._pca.components_[:num_pca_components]

        # ------------------------------------------------------------------------------
        # Neutralize the embeddings provided in words_to_neutralize.
        # if words_to_neutralize is None, the debias will be performed in all embeddings.
        logger.debug("Neutralizing embeddings")
        words_to_neutralize = self._get_words_to_neutralize(
            model, definitional_sets, words_to_neutralize
        )

        # ------------------------------------------------------------------------------
        # Equalize embeddings:

        # Get the equalization sets embeddings.
        # Note that the equalization sets are the same as the definitional sets.
        logger.debug("Obtaining equalize pairs.")
        self._equalize_pairs_embeddings = model.get_embeddings_from_sets(
            sets=definitional_sets,
            sets_name="equalize",
            normalize=True,
            warn_lost_sets=True,
            verbose=verbose,
        )

    def run_debias(
        self,
        word_embedding_model: WordEmbeddingModel,
        definitional_sets: Sequence[Sequence[str]],
        words_to_neutralize: Optional[Sequence[str]] = None,
        pca_args: Dict[str, Any] = {"n_components": 10},
        num_pca_components: int = 1,
        debias_criterion_name: Union[str, None] = None,
        inplace: bool = True,
        verbose: bool = True,
    ) -> WordEmbeddingModel:
        """Execute Bolukbasi's Hard Debias.

        Parameters
        ----------
        word_embedding_model : WordEmbeddingModel
            A word embedding model to debias.

        definitional_sets : Union[List[List[str]], Tuple[Tuple[str]], np.ndarray]
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

        # ------------------------------------------------------------------------------:
        # Normalize embeddings
        logger.debug("Normalizing embeddings.")
        word_embedding_model.normalize_embeddings()

        # ------------------------------------------------------------------------------:
        # Obtain the embedding of the definitional sets.
        logger.debug("Obtaining definitional sets.")
        self._definitional_sets_embeddings = word_embedding_model.get_embeddings_from_sets(
            sets=definitional_sets,
            sets_name="definitional",
            warn_lost_sets=True,
            verbose=verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning sets.
        logger.debug("Identifying the bias subspace.")
        self._pca = self._identify_bias_subspace(
            self._definitional_sets_embeddings, pca_args, verbose,
        )
        self._bias_direction = self._pca.components_[:num_pca_components]

        # ------------------------------------------------------------------------------
        # Neutralize the embeddings provided in words_to_neutralize.
        # if words_to_neutralize is None, the debias will be performed in all embeddings.
        logger.debug("Neutralizing embeddings")
        words_to_neutralize = self._get_words_to_neutralize(
            word_embedding_model, definitional_sets, words_to_neutralize
        )

        self._neutralize_embeddings(
            word_embedding_model, self._bias_direction, words_to_neutralize,
        )

        logger.debug("Normalizing embeddings.")
        word_embedding_model.normalize_embeddings()
        # ------------------------------------------------------------------------------
        # Equalize embeddings:

        # Get the equalization sets embeddings.
        # Note that the equalization sets are the same as the definitional sets.
        logger.debug("Obtaining equalize pairs.")
        self._equalize_pairs_embeddings = word_embedding_model.get_embeddings_from_sets(
            sets=definitional_sets,
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
