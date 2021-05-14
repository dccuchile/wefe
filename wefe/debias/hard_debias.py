"""Bolukbasi et al. Hard Debias WEFE implementation."""
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional, Sequence, Set

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from wefe.word_embedding_model import EmbeddingDict, WordEmbeddingModel
from wefe.debias.base_debias import BaseDebias
from wefe.utils import check_is_fitted


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

    def __init__(self, pca_args: Dict[str, Any] = None) -> None:
        super().__init__()

        if pca_args is None:
            self.pca_args = {"n_components": 10}
        else:
            self.pca_args = pca_args

    def _check_sets_size(
        self, sets: Sequence[Sequence[str]], set_name: str,
    ):

        for idx, set_ in enumerate(sets):
            if len(set_) != 2:
                raise ValueError(
                    f"The {set_name} pair at position {idx} has less/more elements "
                    "than allowed by this method. "
                    f"Number of elements: {len(set_)} - Allowed: 2. "
                    f"Words in the set: {set_}"
                )

    def _identify_bias_subspace(
        self, definning_pairs_embeddings: List[EmbeddingDict], verbose: bool = False,
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

        pca = PCA(**self.pca_args)
        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA variance explained: {explained_variance[0:pca.n_components_]}")

        return pca

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

        # info_log = np.linspace(0, len(embedding_model.vocab), 11, dtype=int)

        for idx, word in tqdm(enumerate(embedding_model.vocab)):
            if word not in bias_criterion_specific_words:
                current_embedding = embedding_model[word]
                neutralized_embedding = self._drop(
                    current_embedding, bias_direction  # type: ignore
                )
                neutralized_embedding = neutralized_embedding.astype(np.float32)
                embedding_model.update_embedding(word, neutralized_embedding)

            # if idx in info_log:
            #     print(
            #         f"Progress: {np.trunc(idx/ len(embedding_model.vocab) * 100)}% "
            #         f"- Current word index: {idx}"
            #     )

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
            embedding_model.update_embedding(word_a, new_a.astype(np.float32))
            embedding_model.update_embedding(word_b, new_b.astype(np.float32))

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        debias_criterion_name: Optional[str] = None,
        verbose: bool = False,
    ) -> "BaseDebias":
        """Compute the debias direction that will be neutralized later.

        Parameters
        ----------
        definitional_pairs : Sequence[Sequence[str]],
            A sequence of string pairs that will be used to define the bias direction.
            For example, for the case of gender debias, this list could be [['woman', 
            'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ...].
        model : WordEmbeddingModel
            The word embedding model to debias.
        verbose: bool, optional
            If true, it prints information about the debias status at each step,
            by default True.

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """
        # ------------------------------------------------------------------------------
        # Check arguments types

        if debias_criterion_name is None or isinstance(debias_criterion_name, str):
            self.debias_criterion_name_ = debias_criterion_name
        else:
            raise ValueError(
                f"debias_criterion_name should be str, got: {debias_criterion_name}"
            )

        self._check_sets_size(definitional_pairs, "definitional")
        self.definitional_pairs_ = definitional_pairs

        # ------------------------------------------------------------------------------
        # Obtain the embedding of each definitional pairs.
        if verbose:
            print("Obtaining definitional pairs.")
        self.definitional_pairs_embeddings_ = model.get_embeddings_from_sets(
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=True,
            normalize=True,
            verbose=verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        if verbose:
            print("Identifying the bias subspace.")

        self.pca_ = self._identify_bias_subspace(
            self.definitional_pairs_embeddings_, verbose,
        )
        self.bias_direction_ = self.pca_.components_[0]

        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        ignore: Optional[List[str]] = None,
        equalize_pairs: Optional[Sequence[Sequence[str]]] = None,
        copy: bool = True,
        verbose: bool = True,
    ) -> WordEmbeddingModel:
        """Execute hard debias over the provided model.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to be debiased
        ignore :  Optional[List[str]]
            List of words that will not be neutralized.
            **Warning**: In case it is None, the debias will be executed on the
            whole vocabulary, causing words that should retain bias information (such
            as `["spokesman", "wife", "himself", "son", "mother", "father",...] in the 
            case of applying gender debias) to lose it,
            by default None.
        equalize_pairs : Optional[Sequence[Sequence[str]]]
            A list with pairs of strings which will be equalized.
            In the case of passing None, the equalization will be done over the word
            pairs passed in definitional_pairs,
            by default None.
        copy : bool
            If True, perform the debiasing on a copy of the model.
            If False, apply the debias on the model provided.
            **WARNING:** Setting copy with True requires at least 2x RAM of the size
            of the model. Otherwise the execution of the debias may rise
            `MemoryError`, by default True.
        verbose : bool
            If true, it prints information about the transform status at each step,
            by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.
        """

        if verbose:
            print(f"Executing Hard Debias on {model.model_name}")

        # ------------------------------------------------------------------------------
        # Check if the method is fitted
        check_is_fitted(
            self,
            [
                "definitional_pairs_",
                "definitional_pairs_embeddings_",
                "pca_",
                "bias_direction_",
            ],
        )
        # ------------------------------------------------------------------------------
        # Check and process arguments
        if ignore is not None and not isinstance(ignore, list):
            raise ValueError(
                f"ignore should be None or a list of strings, got: {ignore}."
            )

        if ignore is None:
            ignore = []
        else:
            for idx, elem in enumerate(ignore):
                if not isinstance(elem, str):
                    raise ValueError(
                        "All elements in ignore list must be strings"
                        f", got: {elem} at index {idx} "
                    )

        # if equalize pairs are none, set the definitional pairs as the pairs
        # to equalize.
        if equalize_pairs is None:
            self.equalize_pairs_ = self.definitional_pairs_
        else:
            self.equalize_pairs_ = equalize_pairs

        self._check_sets_size(self.equalize_pairs_, "equalize")

        # Get the equalization pairs candidates
        self.equalize_pairs_candidates_ = [
            x
            for e1, e2 in self.equalize_pairs_
            for x in [
                (e1.lower(), e2.lower()),
                (e1.title(), e2.title()),
                (e1.upper(), e2.upper()),
            ]
        ]

        # Get the equalization pairs embeddings candidates
        logger.debug("Obtaining equalize pairs.")
        self.equalize_pairs_embeddings_ = model.get_embeddings_from_sets(
            sets=self.equalize_pairs_candidates_,
            sets_name="equalize",
            warn_lost_sets=True,
            normalize=True,
            verbose=verbose,
        )

        if copy:
            logger.warning(
                "copy argument is True. Transform will attempt to create a copy "
                "of the original model. This may fail due to lack of memory."
            )
            model = deepcopy(model)
            print("Model copy created successfully.")
        else:
            logger.warning(
                "copy argument is False. The execution of this method will mutate "
                "the embeddings of the provided model."
            )

        # ------------------------------------------------------------------------------
        # Execute Neutralization

        # Normalize embeddings
        logger.debug("Normalizing embeddings.")
        model.normalize_embeddings()

        # Neutralize the embeddings appointed in ignore word set:
        logger.debug("Neutralizing embeddings")
        self._neutralize_embeddings(
            model, self.bias_direction_, set(ignore),
        )

        logger.debug("Normalizing embeddings.")
        model.normalize_embeddings()
        # ------------------------------------------------------------------------------
        # Execute Equalization:

        logger.debug("Equalizing embeddings..")
        self._equalize_embeddings(
            model, self.equalize_pairs_embeddings_, self.bias_direction_,
        )

        logger.debug("Normalizing embeddings.")
        model.normalize_embeddings()

        # ------------------------------------------------------------------------------
        # # Generate the new KeyedVectors
        if self.debias_criterion_name_ is not None:
            new_model_name = (
                f"{model.model_name}_{self.debias_criterion_name_}_debiased"
            )
        else:
            new_model_name = (
                f"{model.model_name}_{self.debias_criterion_name_}_debiased"
            )
        model.model_name = new_model_name

        print("Done!")
        logger.setLevel(logging.INFO)

        return model

