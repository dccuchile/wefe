"""Bolukbasi's Hard Debias WEFE implementation"""
import logging

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Iterable, List, Any, Set, Union

from wefe.word_embedding_model import EmbeddingDict, PreprocessorArgs, WordEmbeddingModel
from wefe.debias.base_debias import BaseDebias


class HardDebias(BaseDebias):
    name = "Bolukbasi's Hard Debias"

    def _get_definning_pairs_embeddings(
        self,
        word_embedding_model: WordEmbeddingModel,
        definning_pairs,
        preprocessor_args,
        secondary_preprocessor_args,
        verbose,
    ) -> List[EmbeddingDict]:
        defining_pairs_embeddings = []
        for pair in definning_pairs:
            (
                not_found_words,
                defining_pair_embeddings,
            ) = word_embedding_model.get_embeddings_from_word_set(
                pair, preprocessor_args, secondary_preprocessor_args
            )
            if len(not_found_words) > 0:
                if verbose is True:
                    logging.warn(
                        f"Lost {not_found_words} when converting {pair} definning set "
                        f"to embedding. Omitting this definning pair."
                    )
            else:
                defining_pairs_embeddings.append(defining_pair_embeddings)

        if len(defining_pairs_embeddings) == 0:
            raise Exception(
                "Failed to convert any defining pair to embedding. "
                "To see more details, activate the verbose mode. "
            )

        return defining_pairs_embeddings

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
        matrix = np.array(matrix)

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
        self, embedding: np.ndarray, gender_direction: np.ndarray
    ) -> np.ndarray:
        return embedding - gender_direction * embedding.dot(
            gender_direction
        ) / gender_direction.dot(gender_direction)

    def _get_neutralized_embeddings(
        self,
        word_embedding_model: WordEmbeddingModel,
        bias_criterion_specific_words: Set[str],
        bias_direction: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        neutralized_embeddings: Dict[str, np.ndarray] = {}

        #  TODO: Agrgear concurrencia a estas operaciones.
        for i, word in enumerate(word_embedding_model.vocab):
            if word not in bias_criterion_specific_words:
                current_embedding = word_embedding_model[word]
                neutralized_embeddings[word] = self._neutralize_embedding(
                    current_embedding, bias_direction
                )

            if i % 100 == 0:
                print(f"Palabra {i}")

        return neutralized_embeddings

    def run_debias(
        self,
        word_embedding_model: WordEmbeddingModel,
        definning_pairs: Iterable[Iterable[str]],
        bias_criterion_specific_words: List[str],
        preprocessor_args: PreprocessorArgs = {
            "strip_accents": False,
            "lowercase": False,
            "preprocessor": None,
        },
        secondary_preprocessor_args: PreprocessorArgs = None,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = True,
    ):
        # TODO: Agregar parámetro inplace, que indique si se modifica el mismo modelo o no.

        self.definning_pairs_embedding_dict = self._get_definning_pairs_embeddings(
            word_embedding_model=word_embedding_model,
            definning_pairs=definning_pairs,
            preprocessor_args=preprocessor_args,
            secondary_preprocessor_args=secondary_preprocessor_args,
            verbose=verbose,
        )

        self.pca = self.identify_bias_subspace(
            word_embedding_model,
            self.definning_pairs_embedding_dict,
            pca_args,
            verbose,
        )

        self.bias_direction = self.pca.components_[0]
        self.bias_criterion_specific_words = set(bias_criterion_specific_words)

        neutralized_embeddings = self._get_neutralized_embeddings(
            word_embedding_model, self.bias_criterion_specific_words
        )

        word_embedding_model.model.add(
            list(neutralized_embeddings.keys()),
            list(neutralized_embeddings.values()),
            replace=True,
        )

        new_embeddings = None
