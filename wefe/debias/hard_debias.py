"""Bolukbasi's Hard Debias WEFE implementation"""
import logging

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Iterable, List, Any, Set, Tuple, Union

from wefe.word_embedding_model import EmbeddingDict, PreprocessorArgs, WordEmbeddingModel
from wefe.debias.base_debias import BaseDebias


class HardDebias(BaseDebias):
    """Hard Debias debiasing implementation based on Bolukbasi's code.
    https://github.com/tolga-b/debiaswe"""

    name = "Bolukbasi's Hard Debias"

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
                pair, preprocessor_args, secondary_preprocessor_args
            )

            # If some word of the current pair can not be transformed, discard the pair.
            if len(not_found_words) > 0:
                if verbose is True:
                    logging.warn(
                        f"Lost {not_found_words} when converting {pair} {pairs_set_name}"
                        f" pair to embedding. Omitting this definning pair."
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

    def _neutralize_embedding(
        self, embedding: np.ndarray, bias_direction: np.ndarray
    ) -> np.ndarray:

        return embedding - bias_direction * embedding.dot(
            bias_direction
        ) / bias_direction.dot(bias_direction)

    def _neutralize_embeddings(
        self,
        word_embedding_model: WordEmbeddingModel,
        bias_criterion_specific_words: Set[str],
        bias_direction: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:

        words: List[str] = []
        embeddings = []

        #  TODO: Agrgear concurrencia a estas operaciones.
        for i, word in enumerate(word_embedding_model.vocab):
            words.append(word)
            current_embedding = word_embedding_model[word]
            if word not in bias_criterion_specific_words:
                neutralized_embedding = self._neutralize_embedding(
                    current_embedding, bias_direction
                )
                embeddings.append(neutralized_embedding)
            else:
                embeddings.append(current_embedding)

            if i % 10000 == 0:
                logging.info(f"word {i}")

        return words, np.array(embeddings)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        return embeddings

    def _equalize_embeddings(
        self,
        words: List[str],
        embeddings: np.ndarray,
        equalize_pairs_embeddings: List[EmbeddingDict],
        bias_direction: np.ndarray,
    ):

        for equalize_pair_embeddings in equalize_pairs_embeddings:
            (word_a, embedding_a), (
                word_b,
                embedding_b,
            ) = equalize_pair_embeddings.items()

            y = self._neutralize_embedding(
                (embedding_a + embedding_b) / 2, bias_direction
            )
            z = np.sqrt(1 - np.linalg.norm(y) ** 2)
            if (embedding_a - embedding_b).dot(bias_direction) < 0:
                z = -z

            new_a = z * bias_direction + y
            new_b = -z * bias_direction + y

            index_word_a = words.index(word_a)
            index_word_b = words.index(word_b)

            embeddings[index_word_a] = new_a
            embeddings[index_word_b] = new_b

        return words, embeddings

    def _update_embeddings(
        self,
        word_embedding_model: WordEmbeddingModel,
        words: Iterable[str],
        new_embeddings: np.ndarray,
    ):
        if hasattr(word_embedding_model.model, "add_vectors"):
            word_embedding_model.model.add_vectors(
                list(words),
                list(new_embeddings),
                replace=True,
            )
        # With gensim 3
        else:
            word_embedding_model.model.add(
                list(words),
                list(new_embeddings),
                replace=True,
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
        secondary_preprocessor_args: Union[PreprocessorArgs, None] = {
            "strip_accents": True,
            "lowercase": True,
            "preprocessor": None,
        },
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = True,
    ):
        """Execute Bolukbasi's Debias in the word embedding model provided.
        WARNING: Requires 2x RAM of the size of the model. Otherwise it will rise
        MemoryError.

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

        # TODO: Agregar parámetro inplace, que indique si se modifica el mismo
        # modelo o no.

        # Obtain the embedding of the definitional pairs from the word sets.
        self._definitional_pairs_embeddings = self._get_embeddings_from_pairs_sets(
            word_embedding_model=word_embedding_model,
            pairs=definitional_pairs,
            preprocessor_args=preprocessor_args,
            secondary_preprocessor_args=secondary_preprocessor_args,
            verbose=verbose,
            pairs_set_name="definitional",
        )

        self._equalize_pairs_embeddings = self._get_embeddings_from_pairs_sets(
            word_embedding_model=word_embedding_model,
            equalize_pairs=equalize_pairs,
            preprocessor_args=preprocessor_args,
            secondary_preprocessor_args=secondary_preprocessor_args,
            verbose=verbose,
            pairs_set_name="equalize",
        )

        # Identifies the bias subspace from the definning pairs
        self._pca = self._identify_bias_subspace(
            self._definitional_pairs_embeddings,
            pca_args,
            verbose,
        )

        self._bias_direction = self._pca.components_[0]
        self.bias_criterion_specific_words = set(bias_criterion_specific_words)

        words, neutralized_embeddings = self._neutralize_embeddings(
            word_embedding_model, self.bias_criterion_specific_words.self._bias_direction
        )

        words, neutralized_embeddings = self._normalize_embeddings(
            neutralized_embeddings
        )
        # Use add function to replace the unbiased embeddings with the new ones.
        # With gensim 4:

        # Clean the new embeddings
        words, equalized_embeddings = self._equalize_embeddings(
            words,
            neutralized_embeddings,
            self._equalize_pairs_embeddings,
            self._bias_direction,
        )
        equalized_embeddings = self._normalize_embeddings(equalized_embeddings)
        self._update_embeddings(word_embedding_model, words, equalized_embeddings)
