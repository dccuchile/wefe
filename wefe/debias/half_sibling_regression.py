"""Half Sibling Regression WEFE implementation."""

from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from wefe.debias.base_debias import BaseDebias
from wefe.preprocessing import get_embeddings_from_tuples
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel


class HalfSiblingRegression(BaseDebias):
    r"""Half Sibling Debias method.

    This method proposes to learn spurious gender information via causal
    inference by utilizing the statistical dependency between gender-biased
    word vectors and gender definition word vectors. The learned spurious
    gender information is then subtracted from the gender-biased word
    vectors to achieve gender-debiasing as the following where :math:`V_n` are
    the debiased word vectors, Vn are non gender definition and :math:`G` is
    the approximated gender information:

    .. math::

        V_n' := V_n - G

    G is obtained by predicting Non gender definition word vectors (:math:`V_n`)
    using the gender-definition word vectors (:math:`V_d`):

    .. math::

        G := E[V_n|V_d]

    The Prediction is done by a Ridge Regression following the next steps:

    1. Compute the weight matrix of a Ridge Regression using two sets of words

    .. math::

        W = ((V_d)^T V_d +  \alpha I)^{-1} (V_d)^TV_n

    2. Compute the gender information:

    .. math::

        G = V_d W

    3. Subtract gender information from non gender definition words:

    .. math::

        V_n' = V_n - G

    This method is binary because it only allows 2 classes of the same bias
    criterion, such as male or female.
    For a multiclass debias (such as for Latinos, Asians and Whites), it
    is recommended to visit MulticlassHardDebias class.

    .. warning::

        This method requires three times the memory of the model when a copy of
        the model is made and two times the memory of the model if not. Make sure this
        much memory is available.

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
    >>> # load gender specific words, in this case the ones included in wefe
    >>> debiaswe_wordsets = fetch_debiaswe()
    >>> gender_specific = debiaswe_wordsets["gender_specific"]
    >>>
    >>> # instance and fit the method
    >>> hsr = HalfSiblingRegression().fit(
    ...     model=model, definitional_words=gender_specific
    ... )
    >>> # execute the debias on the words not included in the gender definition set
    >>> debiased_model = hsr.transform(model = model)
    Copy argument is True. Transform will attempt to create a copy of the original
    model. This may fail due to lack of memory.
    Model copy created successfully.
    >>>
    >>>
    >>> # if you want the debias over a specific set of words  you can
    >>> #include them in the target parameter
    >>> debiased_model = hsr.transform(
    ...     model=model, target=["doctor", "nurse", "programmer"]
    ... )
    Copy argument is True. Transform will attempt to create a copy of the original
    model. This may fail due to lack of memory.
    Model copy created successfully.
    >>>
    >>> # if you want to exclude a set of words from the debias process
    >>> # you can include them in the ignore parameter
    >>> debiased_model = hsr.transform(
    ...     model=model, ignore=["dress", "beard", "niece", "nephew"]
    ... )
    Copy argument is True. Transform will attempt to create a copy of the original
    model. This may fail due to lack of memory.
    Model copy created successfully.

    References
    ----------
    | [1]: Yang, Zekun y Juan Feng: A causal inference method for reducing
    |      gender bias in word embedding relations.
    |      In Proceedings of the AAAI Conference on Artificial Intelligence,
           volumen 34, pages 9434–9441, 2020
    | [2]: https://github.com/KunkunYang/GenderBiasHSR
    | [3]: Bernhard Sch ̈olkopf, David W. Hogg, Dun Wang,
    |      Daniel Foreman-Mackey, Dominik Jan-zing, Carl-Johann Simon-Gabriel,
           and Jonas Peters.
    |      Modeling confounding by half-sibling regression.
    |      Proceedings of the National Academy of Sciences, 113(27):7391–7398, 2016

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

    def _get_bias_vectors(
        self, model: WordEmbeddingModel, bias_definitional_words: list[str]
    ) -> np.ndarray:
        vectors = [model[word] for word in bias_definitional_words if word in model]
        return np.asarray(vectors)

    def _get_non_bias_dict(
        self, model: WordEmbeddingModel, non_bias: list[str]
    ) -> dict[str, np.ndarray]:
        dictionary = get_embeddings_from_tuples(
            model=model, sets=[non_bias], sets_name="non_bias", normalize=False
        )
        return dictionary[0]

    def _compute_weigth_matrix(
        self, bias_vectors: np.ndarray, non_bias_vectors: np.ndarray, alpha: float
    ) -> np.ndarray:
        a = bias_vectors.T @ bias_vectors + alpha * np.eye(bias_vectors.shape[1])
        b = bias_vectors.T @ non_bias_vectors
        weight_matrix = np.linalg.inv(a) @ b
        return weight_matrix

    def _compute_bias_information(
        self, bias_vectors: np.ndarray, weight_matrix: np.ndarray
    ) -> np.ndarray:
        bias_information = bias_vectors @ weight_matrix
        return bias_information

    def _subtract_bias_information(
        self, non_bias_vectors: np.ndarray, bias_information: np.ndarray
    ) -> np.ndarray:
        debiased_vectors = non_bias_vectors - bias_information
        return debiased_vectors

    def _get_indexes(
        self,
        model: WordEmbeddingModel,
        target: list[str],
        non_bias: list[str],
    ) -> list[int]:
        return [non_bias.index(word) for word in target if word in model]

    def fit(
        self,
        model: WordEmbeddingModel,
        definitional_words: list[str],
        alpha: float = 60,
    ) -> BaseDebias:
        """Compute the weight matrix and the bias information.

        Parameters
        ----------
        model: WordEmbeddingModel
            The word embedding model to debias.
        definitional_words: List[str]
            List of strings. This list contains words that embody bias
            information by definition.
        alpha: float
            Ridge Regression constant. By default 60.

        Returns
        -------
        BaseDebias
            The debias method fitted.

        """
        self.bias_definitional_words = definitional_words
        self.non_bias = list(
            set(model.vocab.keys()) - set(self.bias_definitional_words)
        )
        self.alpha = alpha

        bias_definitional_words_vectors = self._get_bias_vectors(
            model, self.bias_definitional_words
        ).T

        self.non_bias_dict = self._get_non_bias_dict(model, self.non_bias)

        # ------------------------------------------------------------------------------
        # Compute the weight matrix .
        if self.verbose:
            print("Computing the weight matrix.")
        weigth_matrix = self._compute_weigth_matrix(
            bias_definitional_words_vectors,
            np.asarray(list(self.non_bias_dict.values())).T,
            alpha=self.alpha,
        )

        # ------------------------------------------------------------------------------:
        # Compute the approximated bias information
        if self.verbose:
            print("Computing bias information")
        self.bias_information = self._compute_bias_information(
            bias_definitional_words_vectors, weigth_matrix
        )

        return self

    def transform(
        self,
        model: WordEmbeddingModel,
        target: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
        copy: bool = True,
    ) -> WordEmbeddingModel:
        """Substracts the gender information from vectors.

        Parameters
        ----------
        model : WordEmbeddingModel
            The word embedding model to mitigate.
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

        Returns
        -------
        WordEmbeddingModel
            The debiased embedding model.

        """
        # check if the following attributes exist in the object.
        check_is_fitted(
            self,
            ["bias_definitional_words", "non_bias", "alpha", "non_bias_dict"],
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
        # Substract bias information from vectors:

        if self.verbose:
            print("Subtracting bias information.")
        # if target or ignore are specified the debias is applied only in the
        # columns corresponding to those words embeddings
        if target or ignore:
            if target:
                target = target

            elif ignore:
                target = list(set(self.non_bias_dict.keys()) - set(ignore))

            indexes = self._get_indexes(model, target, list(self.non_bias_dict.keys()))

            bias_info = self.bias_information[:, indexes]
            vectors = np.asarray(list(self.non_bias_dict.values())).T[:, indexes]
            debiased_vectors = self._subtract_bias_information(vectors, bias_info).T
            self.non_bias_dict = dict(zip(target, debiased_vectors))

        # if target and ignores are not provided the debias is applied to
        # all non bias vectors
        else:
            vectors = np.asarray(list(self.non_bias_dict.values())).T
            debiased_vectors = self._subtract_bias_information(
                vectors, self.bias_information
            ).T
            self.non_bias_dict = dict(zip(self.non_bias_dict.keys(), debiased_vectors))

        if self.verbose:
            print("Updating debiased vectors")

        # -------------------------------------------------------------------
        # update the model with new vectors
        for word in tqdm(self.non_bias_dict.keys()):
            model.update(word, self.non_bias_dict[word].astype(model.wv.vectors.dtype))

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
