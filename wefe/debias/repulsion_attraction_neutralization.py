from copy import deepcopy
import types
from typing import Dict, Any, Optional, List, Sequence

from tqdm import tqdm
from wefe.debias.base_debias import BaseDebias
from sklearn.decomposition import PCA
from wefe.preprocessing import EmbeddingDict, get_embeddings_from_sets
import numpy as np
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel

try:
    import torch.nn as nn
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch is required to run RepulsionAttractionNeutralization method.\
        Visit https://pytorch.org/ to install it."
    )


class RAN(nn.Module):
    """Class to perform the optimization by gradient descent of the
    objective function.
    """

    def __init__(
        self,
        model: WordEmbeddingModel,
        word: str,
        w_b: np.array,
        w: np.array,
        repulsion_set: List[np.ndarray],
        bias_direction: np.array,
        objective_function: types.FunctionType,
        weights=[0.33, 0.33, 0.33],
    ):
        """Initialize a RAN instance
        Parameters
        ----------
            model: WordEmbeddingModel
                The word embedding model to debias.
            word: str
                Word to be debiased
            w_b: np.array
                Debiased embedding of word
            w: np.array
                Original embedding of word
            repulsion_set: List[np.ndarray]
                Set of embeddings to be repeled from word
            bias_direction: np.array

            objective_function: types.FunctionType
                Function to be minimized to obtain the debiased embedding
            weights: list, optional
                weights λi that determine the relative importance of one
                objective function (repulsion, attarction, neutralization)
                over another. by Defaults [0.33, 0.33, 0.33].
        """
        super(RAN, self).__init__()

        self.model = model
        self.word = word
        self.w = torch.FloatTensor(np.array(w)).requires_grad_(True)
        if len(repulsion_set) == 0:
            self.repulsion_set = False
        else:
            self.repulsion_set = torch.FloatTensor(
                np.array(repulsion_set)
            ).requires_grad_(True)

        self.w_b = nn.Parameter(w_b)

        self.bias_direction = torch.FloatTensor(np.array(bias_direction))

        self.weights = weights

        self.objective_function = objective_function

    def forward(self):
        return self.objective_function(
            self.w_b, self.w, self.bias_direction, self.repulsion_set, self.weights
        )


class RepulsionAttractionNeutralization(BaseDebias):
    """Repulsion Attraction Neutralization method.

    This method allow reducing the bias of an embedding model creating a
    transformation such that the stereotypical gender information are
    minimized with minimal semantic offset. This transformation bases
    its operations on:

    1. Repelling embeddings from neighbours with a high value of indirect
    bias (indicating a strong association due to gender), to minimize the
    gender bias based illicit associations.
    2. Attracting debiased embeddings to the original represention, to
    minimize the loss of semantic meaning
    3. Neutralizing the gender direction of each word, minimizing its
    bias to any particular gender.

    This method is binary because it only allows 2 classes of the same bias
    criterion,such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites), it
    is recommended to visit MulticlassHardDebias class.

    The steps followed to perform the debias are:

    1. **Identify a bias subspace through the defining sets.** In the case of
    gender, these could be e.g. `{'woman', 'man'}, {'she', 'he'}, ...`

    2. A multi-objective optimization is performed. For each vector w in the
    target set it is found its debias counterpart wd by solving:

    argmin(Fr(wd),Fa(wd),Fn(wd))

    where Fr, Fa, Fn are repulsion, attraction and neutralization functions
    defined as the following:

    Fr(wd) =  Σ |cos(wd,ni)| / |S|
    Fa(wd) = |cos(wd,w)-1|/2
    Fn(wd) = |cos(wd,g)|

    The optimization is performed by formulating a single objective:
    F(wd) =  λ1Fr(wd) + λ2Fa(wd) + λ3Fn(wd)

    In the original implementation is define a preserve set (Vp) corresponding
    to words for which gender carries semantic importance, this words are not
    included in the debias process. In WEFE this words would be the ones
    included in the ignore parameter of the transform method. The words
    that are not present in Vp are the ones to be included in the debias
    process and form part of the debias set (Vd), in WEFE this words can
    be specified in the target parameter of the transform method.

        Examples
        --------
        The following example shows how to execute Repulsion Attraction
        Neutralization method that reduces bias in a word embedding model:

        >>> from wefe.debias.repulsion_attraction_neutralization import RepulsionAttractionNeutralization
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
        >>> ran = RepulsionAttractionNeutralization().fit(model = model, definitional_pairs= definitional_pairs)
        >>> # execute the debias passing words over a set of target words
        >>> debiased_model = ran.transform(model = model, target = ['doctor','nurse','programmer'])
        >>>
        >>>
        >>> # if you don't want a set of words to be debiased include them in the ignore set
        >>> gender_specific = debiaswe_wordsets["gender_specific"]
        >>> debiased_model = ran.transform(model = model, ignore= gender_specific)

    References
    ----------
    | [1]: Kumar, Vaibhav, Tenzin Singhay Bhotia y Tanmoy Chakraborty: Nurse
        is Closer to Woman than Surgeon? Mitigating Gender-Biased Proximities
        in Word Embeddings. CoRR,abs/2006.01938, 2020.
        https://arxiv.org/abs/2006.01938
    | [2]: https://github.com/TimeTraveller-San/RAN-Debias
    """

    name = "Repulsion attraction Neutralization"
    short_name = "RAN"

    def __init__(
        self,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = False,
        criterion_name: Optional[str] = None,
    ) -> None:
        """Initialize a Repulsion Attraction Neutralization Debias instance.

        Parameters
        ----------
        pca_args : Dict[str, Any], optional
            Arguments for the PCA that is calculated internally in the
            identification of the bias subspace,
            by default {"n_components": 10}
        verbose : bool, optional
            True will print informative messages about the debiasing process,
            by default False.
        criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g., 'Gender'. This will indicate the name of the model returning
            transform, by default None
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

    def _identify_bias_subspace(
        self,
        defining_pairs_embeddings: List[EmbeddingDict],
        verbose: bool = False,
    ) -> PCA:

        matrix = []
        for embedding_dict_pair in defining_pairs_embeddings:

            # Get the center of the current defining pair.
            pair_embeddings = np.array(list(embedding_dict_pair.values()))
            center = np.mean(pair_embeddings, axis=0)
            # For each word, embedding in the defining pair:
            for embedding in embedding_dict_pair.values():
                # Subtract the center of the pair to the embedding
                matrix.append(embedding - center)
        matrix = np.array(matrix)  # type: ignore

        pca = PCA(**self.pca_args)
        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA variance explained: {explained_variance[0:pca.n_components_]}")

        return pca

    def _check_sets_size(
        self,
        sets: Sequence[Sequence[str]],
        set_name: str,
    ):

        for idx, set_ in enumerate(sets):
            if len(set_) != 2:
                adverb = "less" if len(set_) < 2 else "more"

                raise ValueError(
                    f"The {set_name} pair at position {idx} ({set_}) has {adverb} "
                    f"words than allowed by {self.name}: "
                    f"got {len(set_)} words, expected 2."
                )

    def _indirect_bias(
        self, w: np.ndarray, v: np.ndarray, bias_direction: np.ndarray
    ) -> float:
        wv = np.dot(w, v)
        w_orth = w - np.dot(w, bias_direction) * bias_direction
        v_orth = v - np.dot(v, bias_direction) * bias_direction
        cos_wv_orth = np.dot(w_orth, v_orth) / (
            np.linalg.norm(w_orth) * np.linalg.norm(v_orth)
        )
        bias = (wv - cos_wv_orth) / wv
        return bias

    def _get_neighbours(
        self, model: WordEmbeddingModel, word: str, n_neighbours: int
    ) -> List[str]:
        similar_words = model.wv.most_similar(positive=word, topn=n_neighbours)
        similar_words = list(list(zip(*similar_words))[0])
        return similar_words

    def _get_repulsion_set(
        self,
        model: WordEmbeddingModel,
        word: str,
        bias_direction: np.ndarray,
        theta: float,
        n_neighbours: int,
    ) -> List[np.ndarray]:
        """Obtain the embeddings of the words that should be repealed from
        "word". These are the n_neighbours more similar to "word" whose
        indirect bias is greater than theta.

        Parameters
        ----------
            model: WordEmbeddingModel
                The word embedding model to debias.
            word: str
                Word to debias.
            bias_direction: np.ndarray
                The bias subspace
            theta: float
                Threshold to include neighbours in repulsion set.
            n_neighbours: int
                Number of neighbours to be considered.

        Returns
        --------
            List[np.ndarray]
                The embeddings list conforming the repulsion set.
        """
        neighbours = self._get_neighbours(model, word, n_neighbours)
        repulsion_set = []
        for neighbour in neighbours:
            if (
                self._indirect_bias(model[neighbour], model[word], bias_direction)
                > theta
            ):
                repulsion_set.append(model[neighbour])
        return repulsion_set

    def _cosine_similarity(
        self, w: torch.Tensor, set_vectors: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(set_vectors, w) / (set_vectors.norm(dim=1) * w.norm(dim=0))

    def _repulsion(self, w_b: torch.Tensor, repulsion_set: torch.Tensor):
        if not isinstance(repulsion_set, bool):
            cos_similarity = self._cosine_similarity(w_b, repulsion_set)
            repulsion = torch.abs(cos_similarity).mean(dim=0)
        else:
            repulsion = 0
        return repulsion

    def _attraction(self, w_b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        attraction = (
            torch.abs(torch.cosine_similarity(w_b[None, :], w[None, :]) - 1)[0] / 2
        )
        return attraction

    def _neutralization(
        self, w_b: torch.Tensor, bias_direction: torch.Tensor
    ) -> torch.Tensor:
        neutralization = torch.abs(w_b.dot(bias_direction)).mean(dim=0)
        return neutralization

    def _objective_function(
        self,
        w_b: np.ndarray,
        w: np.ndarray,
        bias_direction: np.ndarray,
        repulsion_set: torch.Tensor,
        weights: List[float],
    ) -> torch.Tensor:
        w1, w2, w3 = weights
        return (
            self._repulsion(w_b, repulsion_set) * w1
            + self._attraction(w_b, w) * w2
            + self._neutralization(w_b, bias_direction) * w3
        )

    def _debias(
        self,
        model: WordEmbeddingModel,
        word: str,
        w: np.ndarray,
        w_b: np.ndarray,
        bias_direction: np.ndarray,
        repulsion_set: List[np.ndarray],
        learning_rate: float,
        epochs: int,
        weights: List[float],
    ) -> torch.Tensor:

        ran = RAN(
            model,
            word,
            w_b,
            w,
            repulsion_set,
            bias_direction,
            self._objective_function,
            weights,
        )
        optimizer = torch.optim.Adam(ran.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = ran.forward()
            out.backward()
            optimizer.step()
        debiased_vector = ran.w_b
        return debiased_vector / torch.norm(debiased_vector)

    def _init_vector(self, model: WordEmbeddingModel, word: str) -> torch.Tensor:
        v = deepcopy(model[word])
        return torch.FloatTensor(np.array(v))

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
    ) -> BaseDebias:
        """
        Compute the bias direction.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to debias.
        definitional_pairs : Sequence[Sequence[str]]
            A sequence of string pairs that will be used to define the bias
            direction. For example, for the case of gender debias, this list
            could be [['woman', 'man'], ['girl', 'boy'], ['she', 'he'],
            ['mother', 'father'], ...].
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
        # Identify the bias subspace using the defining pairs.
        if self.verbose:
            print("Identifying the bias subspace.")

        self.pca_ = self._identify_bias_subspace(
            self.definitional_pairs_embeddings_,
            self.verbose,
        )
        self.bias_direction_ = self.pca_.components_[0]
        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[List[str]] = None,
        ignore: Optional[List[str]] = [],
        learning_rate: float = 0.01,
        copy: bool = True,
        epochs: int = 300,
        theta: float = 0.05,
        n_neighbours: int = 100,
        weights: List[float] = [0.33, 0.33, 0.33],
    ) -> WordEmbeddingModel:

        """
        Executes Repulsion Attraction Neutralization Debias over the
        provided model.

        Args:
            model : WordEmbeddingModel
            The word embedding model to debias.
        target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method will
            be performed only on the word embeddings of this set. If `None`
            is provided, the debias will be performed on all words (except
            those specified in ignore). by default `None`.
        ignore : Optional[List[str]], optional
            If target is `None` and a set of words is specified in ignore,
            the debias method will perform the debias in all words except
            those specified in this set, by default `None`.
        copy : bool, optional
            If `True`, the debias will be performed on a copy of the model.
            If `False`, the debias will be applied on the same model delivered,
            causing its vectors to mutate.
            **WARNING:** Setting copy with `True` requires RAM at least 2x of
            the size of the model, otherwise the execution of the debias may
            raise to `MemoryError`, by default True.
        epochs : int, optional
            number of times that the minimization is done. By default 300
         theta: float, optional
            Indirect bias threshold to select neighbours for the repulsion set.
            By default 0.05
        n_neighbours: int, optimal
            Number of neighbours to be consider for the repulsion set.
            By default 100
        weights:
            List of the 3 initial weights to be used. By default [0.33,0.33,0.33]

        WordEmbeddingModel
            The debiased embedding model.
        """
        # check if the following attributes exist in the object.
        check_is_fitted(
            self,
            [
                "bias_direction_",
                "pca_",
                "definitional_pairs_embeddings_",
                "definitional_pairs_",
            ],
        )

        # ------------------------------------------------------------------------------
        # Copy
        if copy:
            print(
                "Copy argument is True. Transform will attempt to create a copy"
                "of the original model. This may fail due to lack of memory."
            )
            model = deepcopy(model)
            print("Model copy created successfully.")

        else:
            print(
                "copy argument is False. The execution of this method will mutate"
                "the original model."
            )

        # Normalize embeddings
        if self.verbose:
            print("Normalizing embeddings.")
        model.normalize()

        if self.verbose:
            print(
                f"Executing Repulsion attraction Neutralization Debias on {model.name}"
            )

        # If none target words are provided the debias process is executed
        # over the entire vocabulary
        if not target:
            target = list(model.vocab.keys())

        debiased = {}
        for word in tqdm(target):
            if word in ignore or word not in model:
                continue
            w = model[word]
            w_b = self._init_vector(model, word)
            repulsion = self._get_repulsion_set(
                model, word, self.bias_direction_, theta, n_neighbours
            )
            new_embedding = self._debias(
                model,
                word,
                w,
                w_b,
                self.bias_direction_,
                repulsion,
                learning_rate,
                epochs,
                weights,
            )
            debiased[word] = new_embedding.detach().numpy()

        if self.verbose:
            print("Updating debiased vectors")

        for word in tqdm(debiased):
            model.update(word, debiased[word].astype(model.wv.vectors.dtype))

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
