"""Bolukbasi's Hard Debias WEFE implementation."""
import logging
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Iterable, List, Any, Set, Union

from wefe.word_embedding_model import EmbeddingDict, PreprocessorArgs, WordEmbeddingModel
from wefe.debias.base_debias import BaseDebias


class HardDebias(BaseDebias):
    """Hard Debias debiasing method implementation on WEFE.

    References
    ----------
    Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016).
    Man is to computer programmer as woman is to homemaker? debiasing word embeddings.
    Advances in Neural Information Processing Systems.

    https://github.com/tolga-b/debiaswe
    """

    name = "Bolukbasi's Hard Debias"
    short_name = "HD"

    def _get_embeddings_from_pairs_sets(
        self,
        word_embedding_model: WordEmbeddingModel,
        pairs: Iterable[Iterable[str]],
        preprocessor_args: PreprocessorArgs,
        secondary_preprocessor_args: PreprocessorArgs,
        verbose: bool,
        pairs_set_name: str = "definning",
    ) -> List[EmbeddingDict]:

        definitional_pairs_embeddings: List[EmbeddingDict] = []

        # For each definitional pair:
        for pair in pairs:

            # Transform the pair to a embedding dict.
            # i.e., (word_1, word_2) -> {'word_1': embedding, 'word_2'.: embedding}
            (
                not_found_words,
                defining_pair_embeddings,
            ) = word_embedding_model.get_embeddings_from_word_set(
                pair, preprocessor_args, secondary_preprocessor_args, normalize=True
            )

            # If some word of the current pair can not be transformed, discard the pair.
            if len(not_found_words) > 0:
                if verbose is True:
                    logging.warning(
                        f"Word(s) {not_found_words} when converting {pair} {pairs_set_name}"
                        f" pair to embedding. The pair will be omitted."
                    )
            else:
                # Add the embedding dict to defining_pairs_embeddings
                definitional_pairs_embeddings.append(defining_pair_embeddings)

        if len(definitional_pairs_embeddings) == 0:
            raise Exception(
                f"Failed to convert any {pairs_set_name} pair to embedding. "
                "To see more details, activate the verbose mode. "
            )

        return definitional_pairs_embeddings

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
            for _, embedding in embedding_dict_pair.items():
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
                logging.info(f"PCA variance explaned: {explained_variance[0:10]}")
            else:
                logging.info(f"PCA variance explaned: {explained_variance}")

        return pca

    def _drop(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u - v * u.dot(v) / v.dot(v)

    def _neutralize_embeddings(
        self,
        embedding_model: WordEmbeddingModel,
        bias_direction: np.ndarray,
        bias_criterion_specific_words: Set[str],
    ):
        # words = []
        # neutralized_embeddings = []

        info_log = np.linspace(0, len(embedding_model.vocab), 20, dtype=np.int)

        logging.info("Embedding neutralization started")

        for idx, word in enumerate(embedding_model.vocab):
            if word not in bias_criterion_specific_words:
                current_embedding = embedding_model[word]
                neutralized_embedding = self._drop(current_embedding, bias_direction)
                embedding_model.update_embedding(word, neutralized_embedding)

            if idx in info_log:
                logging.info(
                    f"Progress: {np.trunc(idx/ len(embedding_model.vocab) * 100)}% - Current word index: {idx}"
                )

    def _normalize_embeddings(self, debiased_embeddings: EmbeddingDict):
        """Inefficient implementation of the E.normalize debiaswe for EmbeddingDict.

        Tryies to simulate the original debiaswe function E.normalize() that executes:
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]

        Parameters
        ----------
        debiased_embeddings : EmbeddingDict
            [description]
        """
        for word, curr_embedding in debiased_embeddings.items():
            curr_embedding_norm = np.linalg.norm(curr_embedding)
            debiased_embeddings[word] = curr_embedding / curr_embedding_norm

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

    def _create_new_model(
        self,
        word_embedding_model: WordEmbeddingModel,
        debiased_embeddings: EmbeddingDict,
        debias_criterion_name: Union[str, None],
    ) -> WordEmbeddingModel:

        # Get the new model name if it was specified:
        if debias_criterion_name is not None:
            new_model_name = f"{word_embedding_model.model_name}_debiased"
        else:
            new_model_name = (
                f"{word_embedding_model.model_name}_debiased_{debias_criterion_name}"
            )

        # Get the words and the embeddings

        # TODO: Ver el efecto de esta asignación en el uso de la memoria...
        words = np.array(list(debiased_embeddings.keys()))
        embeddings = np.array(list(debiased_embeddings.values()))

        new_model = KeyedVectors(embeddings.shape[1])

        # Gensim 4
        if hasattr(new_model, "add_vectors"):
            new_model.add_vectors(words, embeddings)

        # Gensim 3
        else:
            new_model.add(words, embeddings)

        return WordEmbeddingModel(
            new_model, new_model_name, word_embedding_model.vocab_prefix
        )

    def run_debias(
        self,
        word_embedding_model: WordEmbeddingModel,
        definitional_pairs: Iterable[Iterable[str]],
        bias_criterion_specific_words: List[str],
        equalize_pairs: Iterable[Iterable[str]],
        preprocessor_args: PreprocessorArgs = {
            "strip_accents": False,
            "lowercase": False,
            "preprocessor": None,
        },
        secondary_preprocessor_args: PreprocessorArgs = None,
        pca_args: Dict[str, Any] = {"n_components": 10},
        debias_criterion_name: Union[str, None] = None,
        verbose: bool = True,
    ) -> WordEmbeddingModel:
        """Execute Bolukbasi's Debias in the word embedding model provided.

        **WARNING:** Requires at least 2x RAM of the size of the model. Otherwise the
        execution of this function probably will rise `MemoryError`.

        Parameters
        ----------
        word_embedding_model : WordEmbeddingModel
            A word embedding model.

        definitional_pairs : Iterable[Iterable[str]]
            A iterable with pairs of words that describes the bias direction. e.g. :
            `[['she', 'he'], ['her', 'his'], ...]`

        bias_criterion_specific_words : List[str]
            A collection of words that is specific to the bias criteria and must not be
            debiased (i.e., it is good that these words are associated with a certain
            social group).
            e.g.:`["spokesman", "wife", "himself", "son", "mother", "father",...]`

        equalize_pairs : Iterable[Iterable[str]]
            [description]
        preprocessor_args : PreprocessorArgs, optional
            [description], by default `{"strip_accents": False, "lowercase": False,
            "preprocessor": None}`

        secondary_preprocessor_args : PreprocessorArgs, optional
            [description], by default None
        pca_args : Dict[str, Any], optional
            [description], by default {"n_components": 10}
        verbose : bool, optional
            [description], by default True
        """
        # ------------------------------------------------------------------------------:
        # Normalize
        word_embedding_model.normalize_embeddings()

        # ------------------------------------------------------------------------------:
        # Obtain the embedding of the definitional pairs.
        self._definitional_pairs_embeddings = self._get_embeddings_from_pairs_sets(
            word_embedding_model,
            definitional_pairs,
            preprocessor_args,
            secondary_preprocessor_args,
            verbose,
            "definitional",
        )
        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        self._pca = self._identify_bias_subspace(
            self._definitional_pairs_embeddings, pca_args, verbose,
        )
        self._bias_direction = self._pca.components_[0]

        # ------------------------------------------------------------------------------
        # Neutralize the embeddings appointed in bias_criterion_specific_words:
        self._neutralize_embeddings(
            word_embedding_model,
            self._bias_direction,
            set(bias_criterion_specific_words),
        )

        word_embedding_model.normalize_embeddings()

        neutralized_embeddings = None
        # ------------------------------------------------------------------------------
        # Equalize the embeddings:

        # Get the equalization pairs candidates
        equalize_pairs_candidates = {
            x
            for e1, e2 in equalize_pairs
            for x in [
                (e1.lower(), e2.lower()),
                (e1.title(), e2.title()),
                (e1.upper(), e2.upper()),
            ]
        }

        # Get the equalization pairs embeddings candidates
        self._equalize_pairs_embeddings = self._get_embeddings_from_pairs_sets(
            word_embedding_model,
            equalize_pairs_candidates,
            preprocessor_args,
            secondary_preprocessor_args,
            verbose,
            pairs_set_name="equalize",
        )

        # Execute the equalization
        self._equalize_embeddings(
            word_embedding_model, self._equalize_pairs_embeddings, self._bias_direction,
        )
        word_embedding_model.normalize_embeddings()

        # # ------------------------------------------------------------------------------
        # # Generate the new KeyedVectors
        # new_model = self._create_new_model(
        #     word_embedding_model, self._new_embeddings, debias_criterion_name
        # )
        word_embedding_model.model_name = word_embedding_model.model_name + "_debiased"

        return word_embedding_model
