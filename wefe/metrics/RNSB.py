from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
from scipy.stats import entropy

from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from .base_metric import BaseMetric


class RNSB(BaseMetric):
    """A implementation of Relative Relative Negative Sentiment Bias (RNSB).

    References
    ----------
    Chris Sweeney and Maryam Najafian.
    A transparent framework for evaluating unintended demographic bias in word
    embeddings.
    In Proceedings of the 57th Annual Meeting of the Association for
    Computational Linguistics, pages 1662–1667, 2019.
    """
    def __init__(self):
        """
        Initialize the properties object.

        Args:
            self: (todo): write your description
        """
        super().__init__(('n', 2), 'Relative Negative Sentiment Bias', 'RNSB')

    def __train_classifier(
            self, attribute_embeddings_dict: List[Dict[str, np.ndarray]],
            classifier: BaseEstimator, classifier_params: dict,
            print_model_evaluation: bool):
        """Train the sentiment classifier from the provided attribute embeddings.

        Parameters
        ----------
        attribute_embeddings_dict : dict[str, np.ndarray]
            dict with the attribute key and embeddings
        classifier : BaseEstimator
            Some scikit classifier that must implement predict_proba function.
        classifier_params : dict
            Classifier parameters
        print_model_evaluation : bool
            Indicates if after the training, the funcion will print the model
            evaluation.
        """

        attribute_0_embeddings = list(attribute_embeddings_dict[0].values())
        attribute_1_embeddings = list(attribute_embeddings_dict[1].values())

        # label the attribute set
        positive_attribute_labels = [1 for embedding in attribute_0_embeddings]
        negative_attribute_labels = [
            -1 for embedding in attribute_1_embeddings
        ]

        attributes_embeddings = np.concatenate(
            (attribute_0_embeddings, attribute_1_embeddings))
        attributes_labels = negative_attribute_labels + positive_attribute_labels

        # split the filtered words in train and test sets.
        # it will repeat until both classes have at least one example:
        num_train_negative_examples = 0
        num_train_positive_examples = 0
        while num_train_negative_examples == 0 or num_train_positive_examples == 0:
            split = train_test_split(attributes_embeddings,
                                     attributes_labels,
                                     test_size=0.33)
            X_embeddings_train, X_embeddings_test, y_train, y_test = split
            num_train_negative_examples = y_train.count(-1)
            num_train_positive_examples = y_train.count(1)

        if classifier_params is None:
            classifier_params = {}

        # if there are a specified classifier, use it.
        if classifier is not None:
            classifier = classifier(**classifier_params)

        # else, train logistic classifier
        else:
            classifier_params['solver'] = 'lbfgs'
            classifier_params['max_iter'] = 2000
            classifier_params['multi_class'] = 'multinomial'
            classifier = LogisticRegression(**classifier_params)

        classifier.fit(X_embeddings_train, y_train)

        # evaluate
        if print_model_evaluation:
            y_pred = classifier.predict(X_embeddings_test)
            print(classifier)
            print(
                "Classification Report\n",
                classification_report(y_test,
                                      y_pred,
                                      labels=classifier.classes_))

        return classifier

    def __calc_rnsb(self, target_embeddings_dict: List[Dict[str, np.ndarray]],
                    classifier: BaseEstimator) -> Tuple[np.float_, dict]:
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
            list(target_dict.values())
            for target_dict in target_embeddings_dict
        ]
        target_words_sets = [
            list(target_dict.keys()) for target_dict in target_embeddings_dict
        ]

        # get the probabilities associated with each target word vector
        probabilities = np.array([
            classifier.predict_proba(target_embeddings)
            for target_embeddings in target_embeddings_sets
        ])

        # extract only the negative sentiment probability for each word
        negative_probabilities = np.array(
            [probability[:, 1] for probability in probabilities])

        # flatten the array
        negative_probabilities = np.concatenate([
            negative_probabilities_arr.flatten()
            for negative_probabilities_arr in negative_probabilities
        ])

        # normalization of the probabilities
        sum_of_negative_probabilities = np.sum(negative_probabilities)
        normalized_negative_probabilities = np.array(
            negative_probabilities / sum_of_negative_probabilities)

        # get the uniform dist
        uniform_dist = np.ones(
            normalized_negative_probabilities.shape[0]
        ) * 1 / normalized_negative_probabilities.shape[0]

        # calc the kl divergence
        kl_divergence = entropy(normalized_negative_probabilities,
                                uniform_dist)

        flatten_target_words = [
            item for sublist in target_words_sets for item in sublist
        ]

        # set the probabilities for each word in a dict.
        negative_sentiment_probabilities = {
            word: prob
            for word, prob in zip(flatten_target_words, negative_probabilities)
        }

        return kl_divergence, negative_sentiment_probabilities

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  classifier: BaseEstimator = None,
                  classifier_params: dict = {},
                  print_model_evaluation: bool = False,
                  num_iterations: int = 1,
                  lost_vocabulary_threshold: float = 0.2,
                  warn_filtered_words: bool = False) -> dict:
        """Calculates the RNSB metric over the provided parameters.
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
            A Query object that contains the target and attribute words for
            be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding
            pretrained model.
        classifier : BaseEstimator, optional
            A scikit-learn compatible classifier with predict_proba function,
            by default None,
        classifier_params : dict, optional
            The parameters that will use the the delivered classifier or the
            default classifier (logit), by default {}
        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed., by default False
        num_iterations : int, optional
            The number of times the classifier will be trained and the scores
            will be calculated. by default 1.
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount of
            words in any word set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered
            words, by default False.

        Returns
        -------
        dict
            A dictionary with the kl-divergence, the negative probabilities
            for all tested target words and the normalized distribution
            of probabilities.
        """

        # standard entry procedure.

        # get the embeddings
        embeddings = self._get_embeddings_from_query(
            query, word_embedding, warn_filtered_words,
            lost_vocabulary_threshold)
        # if there is lost_vocabulary_threshold the allowed limit, return the
        #  default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # get the target and attribute embeddings dicts
        target_embeddings, attribute_embeddings = embeddings

        # create the arrays that will contain the scores for each iteration
        calculated_divergences = []
        calculated_negative_sentiment_probabilities = []

        # calculate the scores for each iteration
        for _ in range(num_iterations):

            # train the logit with the train data.
            trained_classifier = self.__train_classifier(
                attribute_embeddings, classifier, classifier_params,
                print_model_evaluation)

            # get the scores
            divergence, negative_sentiment_probabilities = self.__calc_rnsb(
                target_embeddings, trained_classifier)

            calculated_divergences.append(divergence)
            calculated_negative_sentiment_probabilities.append(
                negative_sentiment_probabilities)

        # aggregate results
        divergence = np.mean(np.array(calculated_divergences))
        negative_sentiment_probabilities = dict(
            pd.DataFrame(calculated_negative_sentiment_probabilities).mean())

        sum_of_prob = np.sum(list(negative_sentiment_probabilities.values()))
        negative_sentiment_distribution = {
            word: prob / sum_of_prob
            for word, prob in negative_sentiment_probabilities.items()
        }
        return {
            'query_name': query.query_name_,
            'result': divergence,
            'negative_sentiment_probabilities':
            negative_sentiment_probabilities,
            'negative_sentiment_distribution': negative_sentiment_distribution
        }
