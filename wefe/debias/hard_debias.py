"""Hard Debias WEFE implementation."""
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


class HardDebias(BaseDebias):
    """Hard Debias debiasing method.

    This method allow reducing the bias of an embedding model through geometric
    operations between embeddings.
    This method is binary because it only allows 2 classes of the same bias criterion,
    such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites), it is recommended
    to visit MulticlassHardDebias class.

    The main idea of this method is:

    1. **Identify a bias subspace through the defining sets.** In the case of gender,
    these could be e.g. `{'woman', 'man'}, {'she', 'he'}, ...`

    2. **Neutralize the bias subspace of embeddings that should not be biased.**
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

    3. **Equalizate the embeddings with respect to the bias direction.**.
    Given an equalization set (set of word pairs such as [she, he], [men, women], ...,
    but not limited to the definitional set) this step executes, for each pair,
    an equalization with respect to the bias direction.
    That is, it takes both embeddings of the pair and distributes them at the same
    distance from the bias direction, such that neither is closer to the bias direction
    than the other.

    References
    ----------
    | [1]: Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016).
    | Man is to computer programmer as woman is to homemaker? debiasing word embeddings.
    | Advances in Neural Information Processing Systems.
    | [2]: https://github.com/tolga-b/debiaswe
    """

    name = "Hard Debias"
    short_name = "HD"

    def __init__(
        self,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = False,
        criterion_name: Optional[str] = None,
    ) -> None:
        """Initialize a Hard Debias instance.

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
            e.g., 'Gender'. This will indicate the name of the model returning transform,
            by default None
        """
        # check verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose should be a bool, got {verbose}.")

        self.pca_args = pca_args
        self.verbose = verbose

        if criterion_name is None or isinstance(criterion_name, str):
            self.criterion_name_ = criterion_name
        else:
            raise ValueError(f"criterion_name should be str, got: {criterion_name}")

    def _check_sets_size(
        self, sets: Sequence[Sequence[str]], set_name: str,
    ):

        for idx, set_ in enumerate(sets):
            if len(set_) != 2:
                adverb = "less" if len(set_) < 2 else "more"

                raise ValueError(
                    f"The {set_name} pair at position {idx} ({set_}) has {adverb} "
                    f"words than allowed by {self.name}: "
                    f"got {len(set_)} words, expected 2."
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

    def _neutralize(
        self,
        model: WordEmbeddingModel,
        bias_direction: np.ndarray,
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
            if word not in ignore_ and word in model.vocab:
                current_embedding = model[word]
                neutralized_embedding = self._drop(
                    current_embedding, bias_direction  # type: ignore
                )
                neutralized_embedding = neutralized_embedding.astype(np.float32)
                model.update(word, neutralized_embedding)

    def _equalize(
        self,
        embedding_model: WordEmbeddingModel,
        equalize_pairs_embeddings: List[EmbeddingDict],
        bias_direction: np.ndarray,
    ):
        for equalize_pair_embeddings in equalize_pairs_embeddings:
            if (
                isinstance(equalize_pair_embeddings, dict)
                and len(equalize_pair_embeddings) == 2
            ):
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
                embedding_model.update(word_a, new_a.astype(np.float32))
                embedding_model.update(word_b, new_b.astype(np.float32))

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        equalize_pairs: Optional[Sequence[Sequence[str]]] = None,
        **fit_params,
    ) -> BaseDebias:
        """Compute the bias direction and obtains the equalize embedding pairs.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        definitional_pairs : Sequence[Sequence[str]]
            A sequence of string pairs that will be used to define the bias direction.
            For example, for the case of gender debias, this list could be [['woman',
            'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ...].
        equalize_pairs : Optional[Sequence[Sequence[str]]], optional
            A list with pairs of strings, which will be equalized.
            In the case of passing None, the equalization will be done over the word
            pairs passed in definitional_pairs,
            by default None.

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """
        # Check arguments types
        self._check_sets_size(definitional_pairs, "definitional")
        self.definitional_pairs_ = definitional_pairs

        # ------------------------------------------------------------------------------
        # Obtain the embedding of each definitional pairs.
        if self.verbose:
            print("Obtaining definitional pairs.")
        self.definitional_pairs_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=self.verbose,
            normalize=True,
            verbose=self.verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        if self.verbose:
            print("Identifying the bias subspace.")

        self.pca_ = self._identify_bias_subspace(
            self.definitional_pairs_embeddings_, self.verbose,
        )
        self.bias_direction_ = self.pca_.components_[0]

        # ------------------------------------------------------------------------------:
        # Obtain the equalization pairs.

        # if equalize pairs are none, set the definitional pairs as the pairs
        # to equalize.
        if equalize_pairs is None:
            self.equalize_pairs_ = self.definitional_pairs_
        else:
            self.equalize_pairs_ = equalize_pairs

        self._check_sets_size(self.equalize_pairs_, "equalize")

        # Get the equalization pairs candidates
        if self.verbose:
            print(
                "Obtaining equalize pairs candidates by creating "
                "pairs with lower(), title() and upper()"
            )
        self.equalize_pairs_candidates_ = [
            x
            for e1, e2 in self.equalize_pairs_
            for x in [
                (e1.lower(), e2.lower()),
                (e1.title(), e2.title()),
                (e1.upper(), e2.upper()),
            ]
        ]

        # Obtain the equalization pairs embeddings candidates
        self.equalize_pairs_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=self.equalize_pairs_candidates_,
            sets_name="equalize",
            warn_lost_sets=self.verbose,
            normalize=True,
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
        """Execute hard debias over the provided model.

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
        # ------------------------------------------------------------------------------
        # Check types and if the method is fitted

        self._check_transform_args(
            model=model, target=target, ignore=ignore, copy=copy,
        )

        # check if the following attributes exist in the object.
        check_is_fitted(
            self,
            [
                "definitional_pairs_",
                "definitional_pairs_embeddings_",
                "pca_",
                "bias_direction_",
            ],
        )

        if self.verbose:
            print(f"Executing Hard Debias on {model.name}")
        # ------------------------------------------------------------------------------
        # Copy
        if copy:
            print(
                "Copy argument is True. Transform will attempt to create a copy "
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
        # Execute Neutralization

        # Normalize embeddings
        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()

        # Neutralize the embeddings:
        if self.verbose:
            print("Neutralizing embeddings")
        self._neutralize(
            model=model,
            bias_direction=self.bias_direction_,
            target=target,
            ignore=ignore,
        )

        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()
        # ------------------------------------------------------------------------------
        # Execute Equalization:

        if self.verbose:
            print("Equalizing embeddings.")

        self._equalize(
            model, self.equalize_pairs_embeddings_, self.bias_direction_,
        )

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
