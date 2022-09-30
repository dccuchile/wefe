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
    r"""Relative Relative Negative Sentiment Bias (RNSB).

    The metric was originally proposed in "A transparent framework for evaluating
    unintended demographic bias in word embeddings" [1].

    This metric is based on measuring bias through word sentiment.
    The main idea is that if there was no bias, all words should be equally negative.
    Therefore, its procedure is based on calculating how negative the words in
    the target sets are.

    For this purpose, RNSB trains a classifier that assigns a probability to each
    word of belonging to the negative class (in the original work the classifier is
    trained using
    :func:`~wefe.datasets.load_bingliu`
    of positive and negative words).
    Then, it generates a probability distribution with the probabilities calculated in
    the previous step and compares them to the uniform distribution
    (case where all words have the same probability of being negative) using
    KL divergence.

    When the negative probability distribution is equal to the uniform one (i.e., there
    is no bias), the KL divergence is 0.

    The following description of the metric is WEFE's adaptation of what was presented
    in the original RNSB work.

    RNSB receives as input queries with two attribute sets :math:`A_1` and
    :math:`A_2` and two or more target sets, thus has a template (tuple of numbers that
    defines the allowed target and attribute sets in the query)
    of the form :math:`s=(N,2)` with :math:`N\geq 2`.

    Given a query :math:`Q=(\{T_1,T_2,\ldots,T_n\},\{A_1,A_2\})` RNSB is
    calculated under the following steps:

    1. First constructs a binary classifier  :math:`C_{(A_1,A_2)}(\cdot)` using
       set :math:`A_1` as training examples for the negative class, and :math:`A_2` as
       training examples for the positive class.

    2. After the training process, this classifier gives for every word :math:`w` a
       probability :math:`C_{(A_1,A_2)}(w)` that can be interpreted as the degree of
       association of :math:`w` with respect to  :math:`A_2` (value
       :math:`1-C_{(A_1,A_2)}(w)` is the degree of association with :math:`A_1`).

    3. Then, the metric constructs a probability distribution :math:`P(\cdot)` over all
       the words :math:`w` in :math:`T_1\cup \cdots \cup T_n`, by computing
       :math:`C_{(A_1,A_2)}(w)` and normalizing it to ensure that
       :math:`\sum_w P(w)=1`.

    4. Finally RNSB is calculated as the distance between :math:`P(\cdot)` and
       the uniform distribution :math:`Y(\cdot)` using the KL-divergence.

    The main idea behind RNSB is that the more that :math:`P(\cdot)` resembles a
    uniform distribution, the less biased the word embedding model is.
    Thus, the optimal value is 0.

    You can see the full paper replication in Previous Studies Replication section.

    References
    ----------
    | [1]: Chris Sweeney and Maryam Najafian. A transparent framework for evaluating
           unintended demographic bias in word embeddings.
    |      In Proceedings of the 57th Annual Meeting of the Association for
           Computational Linguistics, pages 1662–1667, 2019.
    | [2]: https://github.com/ChristopherSweeney/AIFairness/blob/master/python_notebooks/Measuring_and_Mitigating_Word_Embedding_Bias.ipynb

    """

    metric_template = ("n", 2)
    metric_name = "Relative Negative Sentiment Bias"
    metric_short_name = "RNSB"

    def _train_classifier(
        self,
        attribute_embeddings_dict: List[Dict[str, np.ndarray]],
        estimator: BaseEstimator = LogisticRegression,
        estimator_params: Dict[str, Any] = {"max_iter": 10000},
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
            by default LogisticRegression,

        estimator_params : dict, optional
            Parameters that will use the classifier, by default { 'max_iter': 10000, }

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
                    "(with test_size=0.1), the first attribute remained with 0 "
                    "training examples."
                )

            if num_train_negative_examples < 1:
                raise Exception(
                    "After splitting the dataset using train_test_split "
                    "(with test_size=0.1), the second attribute remained with 0 "
                    "training examples."
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
            strengthen the results obtained.
            Note that you cannot specify random_state next to n_iterations as this
            would always produce the same results,
            by default 1.

        random_state : Union[int, None], optional
            Seed that allows making the execution of the query reproducible.
            Warning: if a random_state other than None is provided along with
            n_iterations, each iteration will split the dataset and train a
            classifier associated to the same seed, so the results of each iteration
            will always be the same, by default None.

        holdout: bool, optional
            True indicates that a holdout (split attributes in train/test sets) will
            be executed before running the model training.
            This option allows for evaluating the performance of the classifier
            (can be printed using print_model_evaluation=True) at the cost of training
            the classifier with fewer examples. False disables this functionality.
            Note that holdout divides into 80% train and 20% test, performs a shuffle
            and tries to maintain the original ratio of the classes via stratify param,
            by default True.

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
                specified. True uses 'unicode' by default.
            *   ``preprocessor``: ``Callable``. It receives a function that operates
                on each word. In the case of specifying a function, it overrides the
                default preprocessor (i.e., the previous options stop working).

            A list of preprocessor options allows you to search for several
            variants of the words into the model. For example, the preprocessors
            ``[{}, {"lowercase": True, "strip_accents": True}]``
            ``{}`` allows searching first for the original words in the vocabulary of
            the model. In case some of them are not found,
            ``{"lowercase": True, "strip_accents": True}`` is executed on these words
            and then they are searched in the model vocabulary.

        strategy : str, optional
            The strategy indicates how it will use the preprocessed words: 'first' will
            include only the first transformed word found. 'all' will include all
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
        bias using RNSB.
        Note that by default the RNSB score is returned plus the negative class
        probabilities for each word and its distribution (the above probabilities
        normalized to 1).

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
        >>> RNSB().run_query(query, model)
        {
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.10769060995617141,
            "rnsb": 0.10769060995617141,
            "negative_sentiment_probabilities": {
                "female": 0.5742192708509877,
                "woman": 0.32330898567978306,
                "girl": 0.17573260129841273,
                "sister": 0.15229835340332343,
                "she": 0.3761328719677399,
                "her": 0.35739995104539557,
                "hers": 0.2911542159275662,
                "daughter": 0.11714195753410628,
                "male": 0.4550779245077232,
                "man": 0.39826729589696475,
                "boy": 0.17445392462199483,
                "brother": 0.16517694979156405,
                "he": 0.5044892468050808,
                "him": 0.45426103796811057,
                "his": 0.5013980699813614,
                "son": 0.1509265229834842,
            },
            "negative_sentiment_distribution": {
                "female": 0.11103664779476699,
                "woman": 0.0625181838962096,
                "girl": 0.033981372529543176,
                "sister": 0.02944989742595416,
                "she": 0.07273272658861042,
                "her": 0.06911034599602082,
                "hers": 0.0563004234950174,
                "daughter": 0.022651713275710483,
                "male": 0.08799831316676666,
                "man": 0.07701285503210042,
                "boy": 0.033734115115920706,
                "brother": 0.031940228635376634,
                "he": 0.09755296914839981,
                "him": 0.08784035200525282,
                "his": 0.09695522899987082,
                "son": 0.029184626894479145,
            },
        }

        If you want to perform a holdout to evaluate (defualt option) the model
        and print the evaluation, use the params ``holdout=True`` and
        ``print_model_evaluation=True``

        >>> RNSB().run_query(
        ...     query,
        ...     model,
        ...     holdout=True,
        ...     print_model_evaluation=True)
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
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.09400726375514418,
            "rnsb": 0.09400726375514418,
            "negative_sentiment_probabilities": {
                "female": 0.5583010801302075,
                "woman": 0.3159147912504866,
                "girl": 0.20753840501109977,
                "sister": 0.16020059726421976,
                "she": 0.4266765171984158,
                "her": 0.4066467259229203,
                "hers": 0.32435655424005905,
                "daughter": 0.13318012193912765,
                "male": 0.44129601598998147,
                "man": 0.42681869843678866,
                "boy": 0.21830517614567535,
                "brother": 0.2037443178553553,
                "he": 0.5655603842644314,
                "him": 0.512466010254818,
                "his": 0.5689713390373838,
                "son": 0.18364286185769785,
            },
            "negative_sentiment_distribution": {
                "female": 0.09875108690481095,
                "woman": 0.05587832464521873,
                "girl": 0.0367089439708001,
                "sister": 0.02833593497428311,
                "she": 0.07546961904547295,
                "her": 0.07192679290859644,
                "hers": 0.05737148541506475,
                "daughter": 0.023556611770367462,
                "male": 0.0780554843555203,
                "man": 0.07549476775523993,
                "boy": 0.038613347150078775,
                "brother": 0.03603785404499525,
                "he": 0.1000350969111323,
                "him": 0.09064387893111858,
                "his": 0.10063841921015712,
                "son": 0.032482352007143285,
            },
        }


        If you want to disable the holdout, use the param ``holdout=False``.

        >>> # instance the metric and run the query
        >>> RNSB().run_query(
        ...     query,
        ...     model,
        ...     holdout=False,
        ...     print_model_evaluation=True) # doctest: +SKIP
        "Holdout is disabled. No evaluation was performed."
        {
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.12921977967420623,
            "rnsb": 0.12921977967420623,
            "negative_sentiment_probabilities": {
                "female": 0.5815344717945401,
                "woman": 0.28951392462851366,
                "girl": 0.16744925298532254,
                "sister": 0.1365690846140981,
                "she": 0.40677635339222296,
                "her": 0.3861243765483825,
                "hers": 0.2794312043966708,
                "daughter": 0.1035870893754135,
                "male": 0.45492464330345805,
                "man": 0.4143325974603802,
                "boy": 0.18150440132198242,
                "brother": 0.1607078872193466,
                "he": 0.5656269380025241,
                "him": 0.5025663479575841,
                "his": 0.5657745694122852,
                "son": 0.14803033326417403,
            },
            "negative_sentiment_distribution": {
                "female": 0.10881083995606983,
                "woman": 0.05417091306830872,
                "girl": 0.03133140811261611,
                "sister": 0.025553423794525788,
                "she": 0.0761118709786697,
                "her": 0.07224768225706711,
                "hers": 0.05228433658715302,
                "daughter": 0.019382166922557734,
                "male": 0.08512089128923325,
                "man": 0.07752571883094242,
                "boy": 0.03396126510372403,
                "brother": 0.030070032034284336,
                "he": 0.10583438336150637,
                "him": 0.09403512449772648,
                "his": 0.1058620066555313,
                "son": 0.027697936550083888,
            },
        }


        Since each run of RNSB may give a different result due to the random
        ``train_test_split`` and random initializations, RNSB can be requested to
        run many times and returns the average of all runs through the
        parameter ``n_iterations``.
        This makes it potentially more stable and robust to outlier runs.

        >>> RNSB().run_query(query, model, n_iterations=1000)
        {
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.09649701346914233,
            "rnsb": 0.09649701346914233,
            "negative_sentiment_probabilities": {
                "female": 0.5618993210534083,
                "woman": 0.31188456697468364,
                "girl": 0.1968846981458747,
                "sister": 0.1666990161087616,
                "she": 0.4120315698794307,
                "her": 0.3956786125532543,
                "hers": 0.3031550094192968,
                "daughter": 0.13259627603249385,
                "male": 0.45579890258209677,
                "man": 0.4210218238530363,
                "boy": 0.2104231329680286,
                "brother": 0.18879207133574177,
                "he": 0.5473770682025214,
                "him": 0.4924664455586234,
                "his": 0.5479209229372095,
                "son": 0.1770764765027373,
            },
            "negative_sentiment_distribution": {
                "female": 0.10176190651838826,
                "woman": 0.056483371593163176,
                "girl": 0.035656498409823205,
                "sister": 0.030189767202716926,
                "she": 0.07462033949087124,
                "her": 0.07165876247453706,
                "hers": 0.05490241858857015,
                "daughter": 0.024013643264435475,
                "male": 0.08254675451251275,
                "man": 0.07624850551663455,
                "boy": 0.03810835568595322,
                "brother": 0.034190895761652934,
                "he": 0.09913187640146649,
                "him": 0.08918737310881396,
                "his": 0.09923037037111067,
                "son": 0.03206916109934998,
            },
        }

        If you want the embeddings to be normalized before calculating the metrics
        use the ``normalize=True`` before executing the query.

        >>> RNSB().run_query(query, model, normalize=True)
        {
            "query_name": "Female terms and Male Terms wrt Family and Careers",
            "result": 0.00957187793390364,
            "rnsb": 0.00957187793390364,
            "negative_sentiment_probabilities": {
                "female": 0.5078372178028085,
                "woman": 0.4334357574118245,
                "girl": 0.3764103216054252,
                "sister": 0.35256834229924383,
                "she": 0.4454357087596428,
                "her": 0.4390986149718311,
                "hers": 0.41329577968574494,
                "daughter": 0.33427930165282493,
                "male": 0.470250420503012,
                "man": 0.4577545228416623,
                "boy": 0.3698438702135818,
                "brother": 0.35380575403374315,
                "he": 0.49962008627445753,
                "him": 0.47052126448152776,
                "his": 0.49505591114138436,
                "son": 0.34192683607526553,
            },
            "negative_sentiment_distribution": {
                "female": 0.07511118533317328,
                "woman": 0.06410690741777263,
                "girl": 0.05567261405091151,
                "sister": 0.05214628855999085,
                "she": 0.06588174891831232,
                "her": 0.0649444670309599,
                "hers": 0.061128122983393235,
                "daughter": 0.04944126523085684,
                "male": 0.0695519454840733,
                "man": 0.06770375151119586,
                "boy": 0.054701409243182085,
                "brother": 0.05232930677698083,
                "he": 0.07389583823474007,
                "him": 0.06959200440758956,
                "his": 0.07322077821098569,
                "son": 0.050572366605882095,
            },
        }

        RNSB accepts more than 2 sets of target words.
        This example shows how to measure words representing different nationalities
        with respect to positive and negative words.

        Note this is one of the tests that was proposed in the RNSB paper. You can
        see the full paper replication in Previous Studies Replication.

        >>> import gensim.downloader as api
        >>>
        >>> from wefe.word_embedding_model import WordEmbeddingModel
        >>> from wefe.query import Query
        >>> from wefe.datasets import load_bingliu
        >>> from wefe.metrics import RNSB
        >>>
        >>> # Load the model
        >>> model = WordEmbeddingModel(
        ...        api.load('glove-wiki-gigaword-300'), 'Glove wiki'
        ...    )
        >>>
        >>> RNSB_words = [
        ...    ["swedish"],
        ...    ["irish"],
        ...    ["mexican"],
        ...    ["chinese"],
        ...    ["filipino"],
        ...    ["german"],
        ...    ["english"],
        ...    ["french"],
        ...    ["norwegian"],
        ...    ["american"],
        ...    ["indian"],
        ...    ["dutch"],
        ...    ["russian"],
        ...    ["scottish"],
        ...    ["italian"],
        ... ]
        >>>
        >>> bing_liu = load_bingliu()
        >>>
        >>> # Create the query
        >>> query = Query(
        ...     RNSB_words,
        ...     [bing_liu["positive_words"], bing_liu["negative_words"]],
        ...     attribute_sets_names=["Positive words", "Negative words"],
        ... )
        >>>
        >>> results = RNSB().run_query(
        ...     query,
        ...     model,
        ...     preprocessors=[{"lowercase": True}],
        ...     holdout=True,
        ...     print_model_evaluation=True,
        ...     n_iterations=500,
        ... )
        >>> results
        {
            "query_name": "Target set 0, Target set 1, Target set 2, Target set 3, Target set 4, Target set 5, Target set 6, Target set 7, Target set 8, Target set 9, Target set 10, Target set 11, Target set 12, Target set 13 and Target set 14 wrt Positive words and Negative words",
            "result": 0.6313118439654091,
            "rnsb": 0.6313118439654091,
            "negative_sentiment_probabilities": {
                "swedish": 0.03865446713798508,
                "irish": 0.12266387930214015,
                "mexican": 0.5038405165657709,
                "chinese": 0.01913990969357335,
                "filipino": 0.08074140612507152,
                "german": 0.0498762435972975,
                "english": 0.058042779461913364,
                "french": 0.08030917713203162,
                "norwegian": 0.12177903128690087,
                "american": 0.22908203952254658,
                "indian": 0.7836948288757486,
                "dutch": 0.22748838866881654,
                "russian": 0.4877408793080844,
                "scottish": 0.027805085889223837,
                "italian": 0.007885923500742055,
            },
            "negative_sentiment_distribution": {
                "swedish": 0.01361674725376778,
                "irish": 0.04321060837966024,
                "mexican": 0.17748709213331876,
                "chinese": 0.006742385345191273,
                "filipino": 0.02844264587050847,
                "german": 0.017569824481278706,
                "english": 0.020446636995867174,
                "french": 0.028290385255119174,
                "norwegian": 0.04289890438595362,
                "american": 0.08069836330742804,
                "indian": 0.27607092269031136,
                "dutch": 0.08013697047258364,
                "russian": 0.17181569869170976,
                "scottish": 0.009794853090881381,
                "italian": 0.0027779616464207076,
            },
        }


        """
        # check the types of the provided arguments (only the defaults).
        self._check_input(query, model, locals())

        if n_iterations > 1 and random_state is not None:
            raise ValueError(
                "It is not possible to specify random_state together with n_iterations"
                " > 1 since all iterations would produce the same results."
            )

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
