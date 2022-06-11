"""Double Hard Debias WEFE implementation."""
import operator
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from wefe.debias.base_debias import BaseDebias
from wefe.preprocessing import get_embeddings_from_sets
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel


class DoubleHardDebias(BaseDebias):
    """Double Hard Debias Method.

    This method allow reducing the bias of an embedding model through geometric
    operations between embeddings.
    This method is binary because it only allows 2 classes of the same bias
    criterion, such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites),
    it is recommended to visit MulticlassHardDebias class.

    The main idea of this method is:

    1. **Identify a bias subspace through the defining sets.** In the case
    of gender, these could be e.g. `{'woman', 'man'}, {'she', 'he'}, ...`

    2. Find the dominant directions of the entire set of vectors by doing a
    Principal components analysis over it.

    3. Get the target words by finding the most biased words, this is
    the words tha are closest to the representation of each bias group. In
    case of gender 'he' and 'she'.

    3. Try removing each component resulting of PCA and remove also the bias
    direction to every vector in the target set and find which component
    reduces bias the most.

    4. Remove the dominant direction that most reduces bias and remove also
    the bias direction of the vectors in the target set.

    Examples
    --------
    The following example shows how to execute Double Hard Debias method
    that reduces bias in a word embedding model:

    >>> from wefe.debias.double_hard_debias import DoubleHardDebias
    >>> from wefe.utils import load_test_model
    >>> from wefe.datasets import fetch_debiaswe
    >>>
    >>> # load the model (in this case, the test model included in wefe)
    >>> model = load_test_model()
    >>> # load definitional pairs, in this case definitinal pairs included in wefe
    >>> debiaswe_wordsets = fetch_debiaswe()
    >>> definitional_pairs = debiaswe_wordsets["definitional_pairs"]
    >>
    >>> # instance and fit the method
    >>> dhd = DoubleHardDebias().fit(model=model, definitional_pairs=definitional_pairs)
    >>> # execute the debias passing words that represent the bias groups
    >>> debiased_model = dhd.transform(
    ...     model=model, bias_representation=["he", "she"], ignore=gender_specific
    ... )
    >>>
    >>>
    >>> # if you don't want a set of words to be debiased include them in the ignore set
    >>> gender_specific = debiaswe_wordsets["gender_specific"]

    References
    ----------
    | [1]: Wang, Tianlu, Xi Victoria Lin, Nazneen Fatema Rajani, Bryan McCann,
           Vicente Or-donez y Caiming Xiong.
    |      Double-Hard Debias: Tailoring Word Embeddings for GenderBias Mitigation.
    |      CoRR, abs/2005.00965, 2020.https://arxiv.org/abs/2005.00965.
    | [2]: https://github.com/uvavision/Double-Hard-Debias
    """

    name = "Double Hard Debias"
    short_name = "DHD"

    def __init__(
        self,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = False,
        criterion_name: Optional[str] = None,
        incremental_pca: bool = True,
        n_words: int = 1000,
        n_components: int = 4,
    ) -> None:
        """Initialize a Double Hard Debias instance.

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
            e.g., 'Gender'. This will indicate the name of the model
            returning transform, by default None
        incremental_pca: bool, optional
            If `True`, incremental pca will be used over the entire set of
            vectors.
            If `False`, pca will be used over the entire set of vectors.
            **WARNING:** Running pca over the entire set of vectors may
            raise to `MemoryError`,  by default True.
        n_words: int, optional
            Number of target words to be used for each bias group.
            By default 1000
        n_components: int, optional
            Numbers of components of PCA to be used to explore the one that
            reduces bias the most. Usually the best one is close to embedding
            dimension/100. By default 4.
        """
        # check verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose should be a bool, got {verbose}.")

        # check incremental pca
        if not isinstance(incremental_pca, bool):
            raise TypeError(f"incremental_pca should be a bool, got {verbose}.")

        if incremental_pca:
            self.pca_type = IncrementalPCA()
        else:
            self.pca_type = PCA(svd_solver="randomized")

        if not isinstance(pca_args, dict):
            raise TypeError(f"pca_args should be a dict, got {verbose}.")

        self.pca_args = pca_args
        self.verbose = verbose

        if criterion_name is None or isinstance(criterion_name, str):
            self.criterion_name_ = criterion_name
        else:
            raise ValueError(f"criterion_name should be str, got: {criterion_name}")

        if not isinstance(n_words,int):
            raise TypeError(f"n_words should be int, got: {n_words}")
        self.n_words = n_words

        if not isinstance(n_components,int):
            raise TypeError(f"n_components should be int, got: {n_components}")
        self.n_components = n_components

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

    def _similarity(self, u: List[np.ndarray], v: List[np.ndarray]) -> np.array:
        return 1 - pairwise_distances(u, v, metric="cosine")

    def _bias_by_projection(
        self,
        model: WordEmbeddingModel,
        target: List[str],
        ignore: List[str],
        bias_representation: List[str],
    ) -> Dict[str, float]:
        word1 = model[bias_representation[0]]
        word2 = model[bias_representation[1]]
        if target:
            vectors = [model[word] for word in target if word in model]
            words = target
        else:
            vectors = model.wv.vectors
            words = list(model.vocab.keys())
        vectors = model.wv.vectors
        similarities_vectors = (
            self._similarity([word1], vectors)[0]
            - self._similarity([word2], vectors)[0]
        )

        words = list(model.vocab.keys())
        similarities = dict(zip(words, similarities_vectors))
        for word in ignore:
            if word in similarities:
                similarities.pop(word)
        return similarities

    def get_target_words(
        self,
        model: WordEmbeddingModel,
        target: List[str],
        ignore: List[str],
        n_words: int,
        bias_representation: List[str],
    ) -> List[str]:
        """Obtain target words to be debiased.

        This is done by searching the "n_words" most biased words by obtaining the
        words closest to each word in the bias_representation set (in case of gender
        "he" and "she").

        Parameters
        ----------
        model: WordEmbeddingModel
            The word embedding model to debias.
        ignore: List[str]
            Set of words to be ignored from the debias process.
        n_words: int
            number of target words to use.
        bias_representation: Sequence[str]
            Two words that represents de bias groups.

        Returns
        -------
        List[str]
            List of target words for each bias group
        """
        similarities = self._bias_by_projection(model, target, ignore, bias_representation)
        sorted_words = sorted(similarities.items(), key=operator.itemgetter(1))
        female_words = [pair[0] for pair in sorted_words[:n_words]]
        male_words = [pair[0] for pair in sorted_words[-n_words:]]
        return female_words + male_words

    def _principal_components(self, model: WordEmbeddingModel) -> np.ndarray:
        self.pca_type.fit(model.wv.vectors - self.embeddings_mean)
        return self.pca_type.components_

    def _calculate_embeddings_mean(self, model: WordEmbeddingModel) -> float:
        return np.mean(model.wv.vectors, axis=0)

    def _drop_frecuency_features(
        self, components: int, model: WordEmbeddingModel
    ) -> Dict[str, np.ndarray]:
        """Remove the frequency features from the embeddings.

        This is done by removing a component from the ones obtain by the PCA over the
        set of embeddings. The component to remove is indicated by parameter
        "components" and it is removed from the target words embeddings.

        Parameters
        ----------
        components: int
            The number of components to be removed.
        model: WordEmbeddingModel
            A word embedding model.

        Returns
        -------
        Dict[str, np.ndarray]
            The new embeddings for the target words.
        """
        droped_frecuencies = {}

        for word in self.target_words:
            embedding = model[word]
            decendecentralize_embedding = embedding - self.embeddings_mean
            frecuency = np.zeros(embedding.shape).astype(float)

            # for u in self.pca[components]:
            u = self.pca[components]
            frecuency = np.dot(np.dot(np.transpose(u), embedding), u)
            new_embedding = decendecentralize_embedding - frecuency

            droped_frecuencies[word] = new_embedding
        return droped_frecuencies

    def _identify_bias_subspace(
        self, defining_pairs_embeddings, verbose: bool = False,
    ) -> PCA:

        matrix = []
        for embedding_dict_pair in defining_pairs_embeddings:

            # Get the center of the current defining pair.
            pair_embeddings = np.array(list(embedding_dict_pair.values()))
            center = np.mean(pair_embeddings, axis=0)
            # For each word, embedding in the defining pair:
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
        return u - v * u.dot(v) / v.dot(v)

    def _debias(self, words_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        for word in tqdm(words_dict):
            embedding = words_dict[word]
            debias_embedding = self._drop(embedding, self.bias_direction)
            words_dict.update({word: debias_embedding})
        return words_dict

    def _get_optimal_dimension(
        self, model: WordEmbeddingModel, n_words: int, n_components: int
    ) -> int:
        n_components = n_components
        scores = []
        for d in range(n_components):
            result_embeddings = self._drop_frecuency_features(d, model)
            result_embeddings = self._debias(result_embeddings)
            y_true = [0] * n_words + [1] * n_words
            scores.append(self._kmeans_eval(result_embeddings, y_true, n_words))
        min_alignment = min(scores)

        return scores.index(min_alignment)

    def _kmeans_eval(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        y_true: List[int],
        n_words: int,
        n_cluster: int = 2,
    ) -> float:

        embeddings = [
            embeddings_dict[word] for word in self.target_words[0 : 2 * n_words]
        ]
        kmeans = KMeans(n_cluster).fit(embeddings)
        y_pred = kmeans.predict(embeddings)
        correct = [1 if item1 == item2 else 0 for (item1, item2) in zip(y_true, y_pred)]
        alignment_score = sum(correct) / float(len(correct))
        alignment_score = max(alignment_score, 1 - alignment_score)
        return alignment_score

    def fit(
        self, 
        model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        bias_representation: List[str],
    ) -> BaseDebias:
        """Compute the bias direction and obtain the principal components of the entire
           set of vectors.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        definitional_pairs : Sequence[Sequence[str]]
            A sequence of string pairs that will be used to define the bias
            direction. For example, for the case of gender debias, this list
            could be [['woman', 'man'], ['girl', 'boy'], ['she', 'he'],
            ['mother', 'father'], ...].
        bias_representation: List[str]
            Two words that represents each bias group. In case of gender
            "he" and "she".
        Returns
        -------
        BaseDebias
            The debias method fitted.
        """
        self.definitional_pairs = definitional_pairs

        self._check_sets_size(self.definitional_pairs, "definitional")

        # -------------------------------------------------------------------
        # Obtain the embedding of each definitional pairs.
        if self.verbose:
            print("Obtaining definitional pairs.")

        self.definitional_pairs_embeddings = get_embeddings_from_sets(
            model=model,
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=self.verbose,
            normalize=False,
            verbose=self.verbose,
        )
        # -------------------------------------------------------------------
        # Check bias representation
        if bias_representation[0] not in model or bias_representation[1] not in model:
            raise Exception("bias_representation words not in model")
        self.bias_representation = bias_representation

        # -------------------------------------------------------------------
        # Identify the bias subspace using the defining pairs.
        if self.verbose:
            print("Identifying the bias subspace.")
        self.bias_direction = self._identify_bias_subspace(
            self.definitional_pairs_embeddings, verbose=self.verbose
        ).components_[0]

        # -------------------------------------------------------------------
        # Obtain embeddings' mean
        self.embeddings_mean = self._calculate_embeddings_mean(model)

        # -------------------------------------------------------------------
        # Obtain the principal components of all vector in the model.
        if self.verbose:
            print("Obtaining principal components")
        self.pca = self._principal_components(model)

        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        target: List[str] = None,
        ignore: List[str] = [],
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Execute hard debias over the provided model.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        ignore :  List[str], optional
            If set of words is specified in ignore, the debias
            method will perform the debias in all target words except
            those specified in this set, by default [].
        copy : bool, optional
            If `True`, the debias will be performed on a copy of the
            model.
            If `False`, the debias will be applied on the same model delivered,
            causing its vectors to mutate.
            **WARNING:** Setting copy with `True` requires RAM at least 2x of
            the size of the model, otherwise the execution of the debias may
            raise to `MemoryError`, by default True.

        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.
        """
        # check if the following attributes exist in the object.
        self._check_transform_args(
            model=model, ignore=ignore, copy=copy,
        )
        check_is_fitted(
            self,
            [
                "definitional_pairs",
                "definitional_pairs_embeddings",
                "bias_direction",
                "embeddings_mean",
                "pca",
                "bias_representation",
            ],
        )
        if self.verbose:
            print(f"Executing Double Hard Debias on {model.name}")
        # -------------------------------------------------------------------
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
        # -------------------------------------------------------------------
        # Obtain words to apply debias
        if self.verbose:
            print("Obtaining words to apply debias")
            
        if target: 
            self.n_words = len(target) // 2
            
        self.target_words = self.get_target_words(
            model,target, ignore, self.n_words, self.bias_representation
        )

        # -------------------------------------------------------------------
        # Searching best component of pca to debias
        if self.verbose:
            print("Searching component to debias")
        optimal_dimensions = self._get_optimal_dimension(model, self.n_words, self.n_components)

        # -------------------------------------------------------------------
        # Execute debias
        if self.verbose:
            print("Executing debias")
        debiased_embeddings = self._drop_frecuency_features(optimal_dimensions, model)
        debiased_embeddings = self._debias(debiased_embeddings)

        # -------------------------------------------------------------------
        # Update vectors
        if self.verbose:
            print("Updating debiased vectors")
        for word in tqdm(debiased_embeddings):
            model.update(word, debiased_embeddings[word].astype(model.wv.vectors.dtype))
        # -------------------------------------------------------------------
        # # Generate the new KeyedVectors
        if self.criterion_name_ is None:
            new_model_name = f"{model.name}_debiased"
        else:
            new_model_name = f"{model.name}_{self.criterion_name_}_debiased"
        model.name = new_model_name
        if self.verbose:
            print("Done!")
        return model
