from copy import deepcopy
from typing import Dict, Optional, List, Sequence

from tqdm import tqdm
from wefe.debias.base_debias import BaseDebias
from wefe.preprocessing import get_embeddings_from_sets
import numpy as np
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel


class HalfSiblingRegression(BaseDebias):
    """Half Sibling Debias method.
     This method proposes to learn spurious gender information via causal
     inference by utilizing the statistical dependency between gender-biased
     word vectors and gender definition word vectors. The learned spurious
     gender information is then subtracted from the gender-biased word
     vecors to achieve gender-debiasing as the following where Vn' are
     the debiased word vectors, Vn are non gender definition and G is
     the approximated gender information:

     Vn' := Vn - G

     G is obtained by predicting Non gender definition word vectors (Vn)
     using the gender-definition word vectors (Vd):

     G := E[Vn|Vd]

    The Prediction is done by a Ridge Regression following the next steps:

     1. Compute the weight matrix of a Ridge Regression using two sets of words

     W = ((Vd)^T Vd +  αI)^-1 (Vd)^TVn

     2. Compute the gender information:

     G = Vd W

     3. Substract gender information from non gender definition words:

     Vn' = Vn - G

     This method is binary because it only allows 2 classes of the same bias
     criterion, such as male or female.
     For a multiclass debias (such as for Latinos, Asians and Whites), it
     is recommended to visit MulticlassHardDebias class.

     Examples
         --------
         The following example shows how to execute Half Sibling Regression
         Debias method that reduces bias in a word embedding model:

         >>> from wefe.debias.half_sibling_regression import HalfSiblingRegression
         >>> from wefe.utils import load_test_model
         >>> from wefe.datasets import fetch_debiaswe
         >>>
         >>> # load the model (in this case, the test model included in wefe)
         >>> model = load_test_model()
         >>> # load definitional pairs, in this case definitinal pairs included in wefe
         >>> debiaswe_wordsets = fetch_debiaswe()
         >>> gender_specific = debiaswe_wordsets["gender_specific"]
         >>
         >>> # instance and fit the method
         >>> hsr = HalfSiblingRegression().fit(model = model, gender_definition_definition = gender_specific)
         >>> # execute the debias on the words not included in the gender definition set
         >>> debiased_model = hsr.transform(model = model)
         >>>
         >>>
         >>> # if you want the debias over a specific set of words  you can
         >>> #include them in the target parameter
         >>> debiased_model = hsr.transform(model = model, target= ['doctor','nurse','programmer'])
         >>>
         >>> # if you want to excluede a set of words from the debias process
         >>> # you can inlcude them in the ignore parameter
         >>> debiased_model = hsr.transform(model = model, ignore= ['dress','beard','niece','nephew'])

     References
     ----------
     | [1]: Yang, Zekun y Juan Feng: A causal inference method for reducing
     | gender bias in word embedding relations. En Proceedings of the AAAI
     | Conference on Artificial Intelligence,
     | volumen 34, páginas 9434–9441, 2020
     | [2]: https://github.com/KunkunYang/GenderBiasHSR
     | [3]: Bernhard Sch ̈olkopf, David W. Hogg, Dun Wang,
     | Daniel Foreman-Mackey, Dominik Jan-zing, Carl-Johann Simon-Gabriel,
     | and Jonas Peters. Modeling confounding by half-sibling regression.
     | Proceedings of the National Academy of Sciences, 113(27):7391–7398, 2016

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
            e.g., 'Gender'. This will indicate the name of the model returning
            transform, by default None
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

    def _get_indexes(
        self, model, target: List[str], non_gender: List[str]
    ) -> List[int]:
        return [non_gender.index(word) for word in target if word in model]

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
            List of strings. This list contains words that embody gender
            information by definition.

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

            target : Optional[List[str]], optional
            If a set of words is specified in target, the debias method
            will be performed only on the word embeddings of this set.
            If `None` is provided, the debias will be performed on all
            non gender specific words (except those specified in ignore).
            Target words must not be included in the gender specific set.
            by default `None`.

            ignore : Optional[List[str]], optional
            If target is `None` and a set of words is specified in ignore,
            the debias method will perform the debias in all non gender
            specific words except those specified in this
            set, by default `[]`.

            copy : bool, optional
                If `True`, the debias will be performed on a copy of the
                model.
                If `False`, the debias will be applied on the same model
                delivered, causing its vectors to mutate.
                **WARNING:** Setting copy with `True` requires RAM at least
                2x of the size of the model, otherwise the execution of the
                debias may raise to `MemoryError`, by default True.

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
        # Substract gender information from vectors:

        if self.verbose:
            print("Substracting gender information.")
        # if target or ignore are specified the debias is applied only in the
        # columns corresponding to those words embeddings
        if target or ignore:
            if target:
                target = list(set(target) - set(ignore))
            else:
                target = list(set(list(self.non_gender_dict.keys())) - set(ignore))
            indexes = self._get_indexes(
                model, target, list(self.non_gender_dict.keys())
            )

            gender_info = self.gender_information[:, indexes]
            vectors = np.asarray(list(self.non_gender_dict.values())).T[:, indexes]
            debiased_vectors = self._substract_gender_information(
                vectors, gender_info
            ).T
            self.non_gender_dict = dict(zip(target, debiased_vectors))

        # if not target or ignores is provided the debias is applied to
        # all non gender vectors
        else:
            vectors = np.asarray(list(self.non_gender_dict.values())).T
            debiased_vectors = self._substract_gender_information(
                vectors, self.gender_information
            ).T
            self.non_gender_dict = dict(
                zip(self.non_gender_dict.keys(), debiased_vectors)
            )

        if self.verbose:
            print("Updating debiased vectors")

        # -------------------------------------------------------------------
        # update the model with new vectors
        [
            model.update(
                word, self.non_gender_dict[word].astype(model.wv.vectors.dtype)
            )
            for word in tqdm(self.non_gender_dict.keys())
        ]

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
