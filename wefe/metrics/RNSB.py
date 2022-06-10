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


class RNSB(BaseMetric):
    """Relative Relative Negative Sentiment Bias (RNSB).

    The metric was originally proposed in [1].
    Visit `RNSB in Metrics Section <https://wefe.readthedocs.io/en/latest/about.html#rnsb>`_
    for further information.

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
        holdout: bool = True,
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
        # when random_state is not none, set it on classifier params.
        if random_state is not None:
            estimator_params["random_state"] = random_state

        # the attribute 0 words are treated as positive words.
        positive_embeddings = np.array(list(attribute_embeddings_dict[0].values()))
        # the attribute 1 words are treated as negative words.
        negative_embeddings = np.array(list(attribute_embeddings_dict[1].values()))

        # generate the labels (1, -1) for each embedding
        positive_labels = np.ones(positive_embeddings.shape[0])
        negative_labels = -np.ones(negative_embeddings.shape[0])

        attributes_embeddings = np.concatenate(
            (positive_embeddings, negative_embeddings)
        )
        attributes_labels = np.concatenate((positive_labels, negative_labels))

        if holdout:
            split = train_test_split(
                attributes_embeddings,
                attributes_labels,
                shuffle=True,
                random_state=random_state,
                test_size=0.2,
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
        else:
            estimator = estimator(**estimator_params)
            estimator.fit(attributes_embeddings, attributes_labels)
            score = estimator.score(attributes_embeddings, attributes_labels)
            if print_model_evaluation:
                print("Holdout is disabled. No evaluation was performed.")

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
        # row where the negative probabilities are located
        negative_class_clf_row = np.where(classifier.classes_ == -1)[0][0]

        # extract only the negative sentiment probability for each word
        negative_probabilities = np.concatenate(
            [
                probability[:, negative_class_clf_row].flatten()
                for probability in probabilities
            ]
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
        n_iterations: int = 1,
        random_state: Union[int, None] = None,
        holdout: bool = True,
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
        n_iterations parameter.

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

        n_iterations : int, optional
            When provided, it tells the metric to run the specified number of times
            and then average its results. This functionality is indicated to
            strengthen the results obtained, by default 1.

        random_state : Union[int, None], optional
            Seed that allow making the execution of the query reproducible.
            Warning: if a random_state other than None is provided along with
            n_iterations, each iteration will split the dataset and train a
            classifier associated to the same seed, so the results of each iteration
            will always be the same , by default None.

        holdout: bool, optional
            True indicates that a holdout (split attributes in train/test sets) will
            be executed before running the model training.
            This option allows to evaluate the performance of the classifier
            (can be printed using print_model_evaluation=True) at the cost of training
            the classifier with fewer examples. False disables this functionality.
            Note that holdout divides into 80%train and 20% test, performs a shuffle
            and tries to maintain the original ratio of the classes via stratify param.
            by default True

        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed, by default False

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
        The following example shows how to run a query that measures gender
        bias using RNSB:

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
            'query_name': 'Female terms and Male Terms wrt Family and Careers',
            'result': 0.02899395368025491,
            'rnsb': 0.02899395368025491,
            'negative_sentiment_probabilities': {
                'female': 0.43272977959940667,
                'woman': 0.6951544646603257,
                'girl': 0.8141335128074891,
                'sister': 0.8472896023561901,
                'she': 0.5718048693637721,
                'her': 0.5977365245684795,
                'hers': 0.6939932357393684,
                'daughter': 0.8887895021296551,
                'male': 0.5511334216620132,
                'man': 0.584603563015763,
                'boy': 0.8129431089763982,
                'brother': 0.8331301278277582,
                'he': 0.4420145415672582,
                'him': 0.5139776652415698,
                'his': 0.44459083129125154,
                'son': 0.8483699001061482
            },
            'negative_sentiment_distribution': {
                'female': 0.04093015763103808,
                'woman': 0.06575184597373163,
                'girl': 0.07700559236475293,
                'sister': 0.08014169261861909,
                'she': 0.05408470722518866,
                'her': 0.05653747748783378,
                'hers': 0.0656420100321782,
                'daughter': 0.0840670000956609,
                'male': 0.052129478690471215,
                'man': 0.055295283832909777,
                'boy': 0.07689299688658582,
                'brother': 0.07880240525790659,
                'he': 0.04180836566946482,
                'him': 0.04861506614276754,
                'his': 0.04205204648247447,
                'son': 0.0802438736084164
            }
        }

        If you want to perform a holdout to evaluate (defualt option) the model
        and print the evaluation, use the params `holdout=True` and
        `print_model_evaluation=True`

        >>> RNSB().run_query(
        ...     query,
        ...     model,
        ...     holdout=True,
        ...     print_model_evaluation=True) # doctest: +SKIP
        "Classification Report:"
        "              precision    recall  f1-score   support"
        "                                                     "
        "        -1.0       1.00      1.00      1.00         2"
        "         1.0       1.00      1.00      1.00         2"
        "                                                     "
        "    accuracy                           1.00         4"
        "   macro avg       1.00      1.00      1.00         4"
        "weighted avg       1.00      1.00      1.00         4"
        {
            'query_name': 'Female terms and Male Terms wrt Family and Careers',
            'result': 0.028622532697549753,
            'rnsb': 0.028622532697549753,
            'negative_sentiment_probabilities': {
                'female': 0.4253580834091863,
                'woman': 0.7001106999668327,
                'girl': 0.8332271657179001,
                'sister': 0.8396986674252397,
                'she': 0.603565156083575,
                'her': 0.6155296658190583,
                'hers': 0.7147102319731146,
                'daughter': 0.884829695542309,
                'male': 0.5368167185683463,
                'man': 0.5884385611055519,
                'boy': 0.8132056992854114,
                'brother': 0.8270792128939456,
                'he': 0.4500708786239489,
                'him': 0.49965355723589994,
                'his': 0.45394634194580535,
                'son': 0.8450690196299462
            },
            'negative_sentiment_distribution': {
                'female': 0.04000994319670431,
                'woman': 0.0658536664275202,
                'girl': 0.07837483962483958,
                'sister': 0.07898356066672689,
                'she': 0.05677241964432896,
                'her': 0.057897822860029945,
                'hers': 0.06722692455767754,
                'daughter': 0.08322866600691568,
                'male': 0.05049394205657851,
                'man': 0.055349585027011844,
                'boy': 0.07649158463116877,
                'brother': 0.07779655217044128,
                'he': 0.04233447297841125,
                'him': 0.04699830853762932,
                'his': 0.04269900599992016,
                'son': 0.07948870561409564
                }
            }

        If you want to disable the holdout, use the param `holdout=False`.

        >>> # instance the metric and run the query
        >>> RNSB().run_query(
        ...     query,
        ...     model,
        ...     holdout=False,
        ...     print_model_evaluation=True) # doctest: +SKIP
        "Holdout is disabled. No evaluation was performed."
        {
            'query_name': 'Female terms and Male Terms wrt Family and Careers',
            'result': 0.03171747070323668,
            'rnsb': 0.03171747070323668,
            'negative_sentiment_probabilities': {
                'female': 0.41846552820545985,
                'woman': 0.7104860753714863,
                'girl': 0.8325507470146775,
                'sister': 0.8634309153859019,
                'she': 0.593223646607777,
                'her': 0.6138756234516175,
                'hers': 0.7205687956033292,
                'daughter': 0.8964129106245865,
                'male': 0.545075356696542,
                'man': 0.5856674025396198,
                'boy': 0.8184955986780176,
                'brother': 0.8392921127806534,
                'he': 0.43437306199747594,
                'him': 0.4974336520424158,
                'his': 0.4342254305877148,
                'son': 0.851969666735826
            },
            'negative_sentiment_distribution': {
                'female': 0.03927208494188834,
                'woman': 0.0666775818349327,
                'girl': 0.07813308731881921,
                'sister': 0.0810311243458957,
                'she': 0.055672756461026464,
                'her': 0.05761089983046311,
                'hers': 0.06762382332604978,
                'daughter': 0.08412641327954143,
                'male': 0.05115414356760721,
                'man': 0.05496361929467757,
                'boy': 0.07681404203995185,
                'brother': 0.07876574991858241,
                'he': 0.04076497259018534,
                'him': 0.04668307260513937,
                'his': 0.04075111770161401,
                'son': 0.07995551094362546
            }
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
        for i in range(n_iterations):
            try:

                if print_model_evaluation and (i > 0 and i < 2):
                    print(
                        "When n_iterations > 1, only the first evaluation is printed."
                    )
                    print_model_evaluation = False

                # train the logit with the train data.
                trained_classifier, score = self._train_classifier(
                    attribute_embeddings_dict=attribute_embeddings,
                    random_state=random_state,
                    estimator=estimator,
                    estimator_params=estimator_params,
                    holdout=holdout,
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
            except Exception as e:
                logging.exception("RNSB Iteration omitted: " + str(e))

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
            "negative_sentiment_probabilities": negative_sentiment_probabilities,
            "negative_sentiment_distribution": negative_sentiment_distribution,
        }
