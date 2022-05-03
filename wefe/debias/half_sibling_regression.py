from copy import deepcopy
from typing import Dict, Optional, List, Sequence
from wefe.debias.base_debias import BaseDebias
from wefe.preprocessing import get_embeddings_from_sets
import numpy as np
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel


class HalfSiblingRegression(BaseDebias):
    """Half Sibling Debias method.

    This method allow reducing the bias of an embedding model by learning the gender information
    contain in the vectors by perfoming a regression.
    This method is binary because it only allows 2 classes of the same bias criterion,
    such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites), it is recommended
    to visit MulticlassHardDebias class.

    This method is based in a npise elimination method. [3]

    The main idea of this method is:

    1. Compute the weight matrix of a Ridge Regression using two sets of words
    Gender Definition (Vd) and Non gender definition (Vn):
    W = ((Vd)^T Vd +  αI)^-1 (Vd)^TVn

    2. Compute the gender information:
    G = Vd W

    3. Substract gender information from non gender definition words:
    Vn' = Vn - G

    References
    ----------
    | [1]: Yang, Zekun y Juan Feng: A causal inference method for reducing gender bias in word
    | embedding relations. En Proceedings of the AAAI Conference on Artificial Intelligence,
    | volumen 34, páginas 9434–9441, 2020
    | [2]: https://github.com/KunkunYang/GenderBiasHSR
    | [3]: Bernhard Sch ̈olkopf, David W. Hogg, Dun Wang, Daniel Foreman-Mackey, Dominik Jan-
    | zing, Carl-Johann Simon-Gabriel, and Jonas Peters. Modeling confounding by half-sibling
    | regression. Proceedings of the National Academy of Sciences, 113(27):7391–7398, 2016

    """

    name = "Half Sibling Regression"
    short_name = "HSR"

    def __init__(
        self,
        verbose: bool = False,
        criterion_name: Optional[str] = None,
    ) -> None:
        """Initialize a Half Sibling Regression Debias instance.

        Parameters
        ----------
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

        self.verbose = verbose

        if criterion_name is None or isinstance(criterion_name, str):
            self.criterion_name_ = criterion_name
        else:
            raise ValueError(f"criterion_name should be str, got: {criterion_name}")

    def _get_gender_vectors(
        self, model: WordEmbeddingModel, gender_definition: List[str]
    ) -> np.ndarray:

        vectors = [model[word] for word in gender_definition if word in model]
        return np.asarray(vectors)

    def _get_non_gender_dict(
        self, model: WordEmbeddingModel, non_gender: List[str]
    ) -> Dict[str, float]:

        dictionary = get_embeddings_from_sets(
            model=model, sets=[non_gender], sets_name="non_gender", normalize=False
        )
        return dictionary[0]

    def _compute_weigth_matrix(
        self, gender_vectors: np.ndarray, non_gender_vectors: np.ndarray, alpha: float
    ) -> np.ndarray:

        a = gender_vectors.T @ gender_vectors + alpha * np.eye(gender_vectors.shape[1])
        b = gender_vectors.T @ non_gender_vectors
        weight_matrix = np.linalg.inv(a) @ b
        return weight_matrix

    def _compute_gender_information(
        self, gender_vectors: np.ndarray, weight_matrix: np.ndarray
    ) -> np.ndarray:
        gender_information = gender_vectors @ weight_matrix
        return gender_information

    def _substract_gender_information(
        self, non_gender_vectors: np.ndarray, gender_information: np.ndarray
    ) -> np.ndarray:
        debiased_vectors = non_gender_vectors - gender_information
        return debiased_vectors

    def _get_indexes(self, target: List[str], non_gender: List[str]) -> List[int]:
        indexes = []
        for word in target:
            indexes.append(non_gender.index(word))
        return indexes

    def fit(
        self,
        model: WordEmbeddingModel,
        gender_definition: Sequence[str],
        alpha: float = 60,
    ) -> BaseDebias:
        """
        Computes the weight matrix and the gender information

        Parameters
        ----------

        model: WordEmbeddingModel
            The word embedding model to debias.

        gender_definition: Sequence[str]
            List of strings. This list contains words that embody gender information by definition.

        alpha: float
            Ridge Regression constant. By default 60,
            numner

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """

        self.gender_definition = gender_definition
        self.non_gender = list(set(model.vocab.keys()) - set(self.gender_definition))
        self.alpha = alpha

        gender_definition_vectors = self._get_gender_vectors(
            model, self.gender_definition
        ).T

        self.non_gender_dict = self._get_non_gender_dict(model, self.non_gender)

        # ------------------------------------------------------------------------------
        # Compute the weight matrix .
        if self.verbose:
            print("Computing the weight matrix.")
        weigth_matrix = self._compute_weigth_matrix(
            gender_definition_vectors,
            np.asarray(list(self.non_gender_dict.values())).T,
            alpha=self.alpha,
        )

        # ------------------------------------------------------------------------------:
        # Compute the approximated gender information
        if self.verbose:
            print("Computing gender information")
        self.gender_information = self._compute_gender_information(
            gender_definition_vectors, weigth_matrix
        )

        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        target: List[str] = None,
        ignore: List[str] = [],
        copy: bool = True,
    ) -> WordEmbeddingModel:

        """
        Substracts the gender information from vectors.

        Args:
            model: WordEmbeddingModel
                The word embedding model to debias.

            ignore (List[str], optional): _description_. Defaults to [].

            copy : bool, optional
                If `True`, the debias will be performed on a copy of the model.
                If `False`, the debias will be applied on the same model delivered, causing
                its vectors to mutate.
                **WARNING:** Setting copy with `True` requires RAM at least 2x of the size
                of the model, otherwise the execution of the debias may raise to
                `MemoryError`, by default True.

        WordEmbeddingModel
            The debiased embedding model.
        """
        # check if the following attributes exist in the object.
        check_is_fitted(
            self,
            [
                "gender_definition",
                "non_gender",
                "alpha",
                "non_gender_dict",
            ],
        )

        if self.verbose:
            print(f"Executing Half Sibling Debias on {model.name}")

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
        # Substract gender information from vectors:

        if self.verbose:
            print("Substracting gender information.")

        if target:
            target = list(set(target) - set(ignore))
            indexes = self._get_indexes(target, list(self.non_gender_dict.keys()))
            gender_info = self.gender_information[:, indexes]
            vectors = np.asarray(list(self.non_gender_dict.values())).T[:, indexes]
            debiased_vectors = self._substract_gender_information(vectors, gender_info)

        else:
            target = list(set(list(self.non_gender_dict.keys())) - set(ignore))
            indexes = self._get_indexes(target, list(self.non_gender_dict.keys()))
            gender_info = self.gender_information[:, indexes]
            vectors = np.asarray(list(self.non_gender_dict.values())).T[:, indexes]
            debiased_vectors = self._substract_gender_information(vectors, gender_info)

        debiased_vectors = debiased_vectors.T

        for i in range(len(target)):
            self.non_gender_dict[target[i]] = debiased_vectors[
                i
            ]  # update the modified non-stop words

        if self.verbose:
            print("Updating debiased vectors")

        for word in target:
            model.update(
                word, self.non_gender_dict[word].astype(model.wv.vectors.dtype)
            )

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
