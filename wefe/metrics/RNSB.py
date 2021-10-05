"""Relative Negative Sentiment Bias (RNSB) metric implementation."""
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from wefe.metrics.base_metric import BaseMetric
from wefe.preprocessing import get_embeddings_from_query
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

logging.basicConfig(level=logging.DEBUG)


class RNSB(BaseMetric):
    """Relative Relative Negative Sentiment Bias (RNSB).

    References
    ----------
    | [1]: Chris Sweeney and Maryam Najafian. A transparent framework for evaluating
    |      unintended demographic bias in word embeddings.
    |      In Proceedings of the 57th Annual Meeting of the Association for
    |      Computational Linguistics, pages 1662–1667, 2019.
    | [2]: https://github.com/ChristopherSweeney/AIFairness/blob/master/python_notebooks/Measuring_and_Mitigating_Word_Embedding_Bias.ipynb
    """

    metric_template = ("n", 2)
    metric_name = "Relative Negative Sentiment Bias"
    metric_short_name = "RNSB"

    def _train_classifier(
        self,
        attribute_embeddings_dict: List[Dict[str, np.ndarray]],
        estimator: BaseEstimator = LogisticRegression,
        estimator_params: Dict[str, Any] = {"solver": "liblinear", "max_iter": 10000},
        random_state: Union[int, None] = None,
        print_model_evaluation: bool = False,
    ) -> Tuple[BaseEstimator, float]:
        """Train the sentiment classifier from the provided attribute embeddings.

        Parameters
        ----------
        attribute_embeddings_dict : dict[str, np.ndarray]
            A dict with the attributes keys and embeddings

        estimator : BaseEstimator, optional
            A scikit-learn classifier class that implements predict_proba function,
            by default None,

        estimator_params : dict, optional
            Parameters that will use the classifier, by default { 'solver': 'liblinear',
            'max_iter': 10000, }

        random_state : Union[int, None], optional
            A seed that allows making the execution of the query reproducible, by
            default None

        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed, by default False

        Returns
        -------
        Tuple[BaseEstimator, float]
            The trained classifier and the accuracy obtained by the model.
        """
        attribute_0_embeddings = np.array(list(attribute_embeddings_dict[0].values()))
        attribute_1_embeddings = np.array(list(attribute_embeddings_dict[1].values()))

        # generate the labels (1, -1) for each embedding
        positive_attribute_labels = np.ones(attribute_0_embeddings.shape[0])
        negative_attribute_labels = -np.ones(attribute_1_embeddings.shape[0])

        attributes_embeddings = np.concatenate(
            (attribute_0_embeddings, attribute_1_embeddings)
        )
        attributes_labels = np.concatenate(
            (negative_attribute_labels, positive_attribute_labels)
        )

        split = train_test_split(
            attributes_embeddings,
            attributes_labels,
            shuffle=True,
            random_state=random_state,
            test_size=0.1,
            stratify=attributes_labels,
        )
        X_embeddings_train, X_embeddings_test, y_train, y_test = split

        num_train_negative_examples = np.count_nonzero((y_train == -1))
        num_train_positive_examples = np.count_nonzero((y_train == 1))

        # Check the number of train and test examples.
        if num_train_positive_examples == 1:
            raise Exception(
                "After splitting the dataset using train_test_split "
                "(with test_size=0.1), the first attribute remained with 0 training "
                "examples."
            )

        if num_train_negative_examples < 1:
            raise Exception(
                "After splitting the dataset using train_test_split "
                "(with test_size=0.1), the second attribute remained with 0 training "
                "examples."
            )

        # when random_state is not none, set it on classifier params.
        if random_state is not None:
            estimator_params["random_state"] = random_state

        estimator = estimator(**estimator_params)
        estimator.fit(X_embeddings_train, y_train)

        # evaluate
        y_pred = estimator.predict(X_embeddings_test)
        score = estimator.score(X_embeddings_test, y_test)

        if print_model_evaluation:
            print(
                "Classification Report:\n{}".format(
                    classification_report(y_test, y_pred, labels=estimator.classes_)
                )
            )

        return estimator, score

    def _calc_rnsb(
        self,
        target_embeddings_dict: List[Dict[str, np.ndarray]],
        classifier: BaseEstimator,
    ) -> Tuple[np.float_, dict]:
        """Calculate the RNSB metric.

        Parameters
        ----------
        target_embeddings_dict : Dict[str, np.ndarray]
            dict with the target words and their embeddings.
        classifier : BaseEstimator
            Trained scikit-learn classifier in the previous step.

        Returns
        -------
        Tuple[np.float_, dict]
            return the calculated kl_divergence and
            negative_sentiment_probabilities in that order.
        """
        # join the embeddings and the word sets in their respective arrays
        target_embeddings_sets = [
            list(target_dict.values()) for target_dict in target_embeddings_dict
        ]
        target_words_sets = [
            list(target_dict.keys()) for target_dict in target_embeddings_dict
        ]

        # get the probabilities associated with each target word vector
        probabilities = [
            classifier.predict_proba(target_embeddings)
            for target_embeddings in target_embeddings_sets
        ]

        # extract only the negative sentiment probability for each word
        negative_probabilities = np.concatenate(
            [probability[:, 1].flatten() for probability in probabilities]
        )

        # normalization of the probabilities
        normalized_negative_probabilities = np.array(
            negative_probabilities / np.sum(negative_probabilities)
        )

        # get the uniform dist
        uniform_dist = (
            np.ones(normalized_negative_probabilities.shape[0])
            * 1
            / normalized_negative_probabilities.shape[0]
        )

        # calc the kl divergence
        kl_divergence = entropy(normalized_negative_probabilities, uniform_dist)

        flatten_target_words = [
            item for sublist in target_words_sets for item in sublist
        ]

        # set the probabilities for each word in a dict.
        negative_sentiment_probabilities = {
            word: prob
            for word, prob in zip(flatten_target_words, negative_probabilities)
        }

        return kl_divergence, negative_sentiment_probabilities

    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        estimator: BaseEstimator = LogisticRegression,
        estimator_params: Dict[str, Any] = {"solver": "liblinear", "max_iter": 10000},
        num_iterations: int = 1,
        random_state: Union[int, None] = None,
        print_model_evaluation: bool = False,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: List[Dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Calculate the RNSB metric over the provided parameters.

        Note if you want to use with Bing Liu dataset, you have to pass
        the positive and negative words in the first and second place of
        attribute set array respectively.
        Scores on this metric vary with each run due to different instances
        of classifier training. For this reason, the robustness of these scores
        can be improved by repeating the test several times and returning the
        average of the scores obtained. This can be indicated in the
        num_iterations parameter.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to
            be tested.

        model : WordEmbeddingModel
            A word embedding model.

        estimator : BaseEstimator, optional
            A scikit-learn classifier class that implements predict_proba function,
            by default None,

        estimator_params : dict, optional
            Parameters that will use the classifier, by default { 'solver': 'liblinear',
            'max_iter': 10000, }

        num_iterations : int, optional
            When provided, it tells the metric to run the specified number of times
            and then average its results. This functionality is indicated to
            strengthen the results obtained, by default 1.

        random_state : Union[int, None], optional
            Seed that allow making the execution of the query reproducible.
            Warning: if a random_state other than None is provided along with
            num_iterations, each iteration will split the dataset and train a
            classifier associated to the same seed, so the results of each iteration
            will always be the same , by default None.

        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed., by default False

        preprocessors : List[Dict[str, Union[str, bool, Callable]]]
            A list with preprocessor options.

            A ``preprocessor`` is a dictionary that specifies what processing(s) are
            performed on each word before it is looked up in the model vocabulary.
            For example, the ``preprocessor``
            ``{'lowecase': True, 'strip_accents': True}`` allows you to lowercase
            and remove the accent from each word before searching for them in the
            model vocabulary. Note that an empty dictionary ``{}`` indicates that no
            preprocessing is done.

            The possible options for a preprocessor are:

            *   ``lowercase``: ``bool``. Indicates that the words are transformed to
                lowercase.
            *   ``uppercase``: ``bool``. Indicates that the words are transformed to
                uppercase.
            *   ``titlecase``: ``bool``. Indicates that the words are transformed to
                titlecase.
            *   ``strip_accents``: ``bool``, ``{'ascii', 'unicode'}``: Specifies that
                the accents of the words are eliminated. The stripping type can be
                specified. True uses ‘unicode’ by default.
            *   ``preprocessor``: ``Callable``. It receives a function that operates
                on each word. In the case of specifying a function, it overrides the
                default preprocessor (i.e., the previous options stop working).

            A list of preprocessor options allows you to search for several
            variants of the words into the model. For example, the preprocessors
            ``[{}, {"lowercase": True, "strip_accents": True}]``
            ``{}`` allows first to search for the original words in the vocabulary of
            the model. In case some of them are not found,
            ``{"lowercase": True, "strip_accents": True}`` is executed on these words
            and then they are searched in the model vocabulary.

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. all' will include all
            transformed words found, by default "first".

        normalize : bool, optional
            True indicates that embeddings will be normalized, by default False

        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary, by default False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name, the calculated kl-divergence,
            the negative probabilities for all tested target words and
            the normalized distribution of probabilities.

        Examples
        --------
        >>> from wefe.query import Query
        >>> from wefe.utils import load_test_model
        >>> from wefe.metrics import RNSB
        >>>
        >>> # define the query
        >>> query = Query(
        ...     target_sets=[
        ...         ["female", "woman", "girl", "sister", "she", "her", "hers",
        ...          "daughter"],
        ...         ["male", "man", "boy", "brother", "he", "him", "his", "son"],
        ...     ],
        ...     attribute_sets=[
        ...         ["home", "parents", "children", "family", "cousins", "marriage",
        ...          "wedding", "relatives",],
        ...         ["executive", "management", "professional", "corporation", "salary",
        ...          "office", "business", "career", ],
        ...     ],
        ...     target_sets_names=["Female terms", "Male Terms"],
        ...     attribute_sets_names=["Family", "Careers"],
        ... )
        >>>
        >>> # load the model (in this case, the test model included in wefe)
        >>> model = load_test_model()
        >>>
        >>> # instance the metric and run the query
        >>> RNSB().run_query(query, model) # doctest: +SKIP
        {
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.09223875552506647,
            "kl-divergence": 0.09223875552506647,
            "clf_accuracy": 1.0,
            "negative_sentiment_probabilities": {
                "female": 0.5543954373912665,
                "woman": 0.3107589242224508,
                "girl": 0.18710587484907013,
                "sister": 0.1787081823837198,
                "she": 0.4172419154977331,
                "her": 0.4030950036121549,
                "hers": 0.3126640373120572,
                "daughter": 0.14249529344431694,
                "male": 0.4422224610164615,
                "man": 0.4194123616222211,
                "boy": 0.20556697141459176,
                "brother": 0.19801831727151584,
                "he": 0.5577524826493919,
                "him": 0.514179075019818,
                "his": 0.5544435993736733,
                "son": 0.18711536982098712,
            },
            "negative_sentiment_distribution": {
                "female": 0.09926195811727109,
                "woman": 0.05563995884577796,
                "girl": 0.0335004479837668,
                "sister": 0.0319968796973831,
                "she": 0.0747052496243332,
                "her": 0.07217230999250153,
                "hers": 0.05598106059906622,
                "daughter": 0.02551312816774791,
                "male": 0.07917790162647549,
                "man": 0.07509385803950792,
                "boy": 0.03680582257831352,
                "brother": 0.0354542707060297,
                "he": 0.09986302166025017,
                "him": 0.09206140304753956,
                "his": 0.09927058129913385,
                "son": 0.03350214801490194,
            },
        }
        """
        # check the types of the provided arguments (only the defaults).
        self._check_input(query, model, locals())

        # transform query word sets into embeddings
        embeddings = get_embeddings_from_query(
            model=model,
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessors=preprocessors,
            strategy=strategy,
            normalize=normalize,
            warn_not_found_words=warn_not_found_words,
        )

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {
                "query_name": query.query_name,
                "result": np.nan,
                "rnsb": np.nan,
                "score": np.nan,
                "negative_sentiment_probabilities": {},
                "negative_sentiment_distribution": {},
            }

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        # create the arrays that will contain the scores for each iteration
        calculated_divergences = []
        calculated_negative_sentiment_probabilities = []
        scores = []

        # calculate the scores for each iteration
        for _ in range(num_iterations):

            # train the logit with the train data.
            trained_classifier, score = self._train_classifier(
                attribute_embeddings_dict=attribute_embeddings,
                random_state=random_state,
                estimator=estimator,
                estimator_params=estimator_params,
                print_model_evaluation=print_model_evaluation,
            )

            scores.append(score)

            # get the scores
            divergence, negative_sentiment_probabilities = self._calc_rnsb(
                target_embeddings, trained_classifier
            )

            calculated_divergences.append(divergence)
            calculated_negative_sentiment_probabilities.append(
                negative_sentiment_probabilities
            )

        # aggregate results
        divergence = np.mean(np.array(calculated_divergences))
        negative_sentiment_probabilities = dict(
            pd.DataFrame(calculated_negative_sentiment_probabilities).mean()
        )

        sum_of_prob = np.sum(list(negative_sentiment_probabilities.values()))
        negative_sentiment_distribution = {
            word: prob / sum_of_prob
            for word, prob in negative_sentiment_probabilities.items()
        }
        return {
            "query_name": query.query_name,
            "result": divergence,
            "rnsb": divergence,
            "clf_accuracy": np.mean(scores),
            "negative_sentiment_probabilities": negative_sentiment_probabilities,
            "negative_sentiment_distribution": negative_sentiment_distribution,
        }
