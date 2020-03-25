import logging
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from scipy.stats import entropy
from sklearn.base import BaseEstimator

from ..query import Query
from ..word_embedding_model import WordEmbeddingModel
from .base_metric import BaseMetric


class RNSB(BaseMetric):
    """A implementation of Relative Relative Negative Sentiment Bias (RNSB).
    
    References
    ----------
    Chris Sweeney and Maryam Najafian.
    A transparent framework for evaluating unintended demographic bias in word embeddings.
    In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 1662–1667, 2019.
    """

    def __init__(self):
        super().__init__(('n', 2), 'Relative Negative Sentiment Bias', 'RNSB')

    def __train_classifier(self, attribute_0_embeddings, attribute_1_embeddings, classifier: BaseEstimator,
                           classifier_params: dict, print_model_evaluation: bool):

        # label the attribute set
        negative_attribute_labels = [1 for embedding in attribute_0_embeddings]
        positive_attribute_labels = [-1 for embedding in attribute_1_embeddings]

        attributes_embeddings = np.concatenate((attribute_0_embeddings, attribute_1_embeddings))
        attributes_labels = negative_attribute_labels + positive_attribute_labels

        # split the filtered words in train and test sets.
        X_embeddings_train, X_embeddings_test, y_train, y_test = train_test_split(attributes_embeddings,
                                                                                  attributes_labels, test_size=0.33)
        if classifier_params is None:
            classifier_params = {}

        # if there are a specified classifier, use it.
        if not classifier is None:
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
            print("Classification Report\n", classification_report(y_test, y_pred, labels=classifier.classes_))

        return classifier

    def __calc_rnsb(self, classifier, target_embeddings_all_sets: np.ndarray, target_words_all_sets: np.ndarray):
        # get the probabilities associated with each target word vector
        probabilities = np.array(
            [classifier.predict_proba(target_embeddings) for target_embeddings in target_embeddings_all_sets])

        # extract only the negative sentiment probability for each word
        negative_probabilities = np.array([probability[:, 0] for probability in probabilities])

        # flatten the array
        negative_probabilities = np.concatenate(
            [negative_probabilities_arr.flatten() for negative_probabilities_arr in negative_probabilities])

        # normalization of the probabilities
        sum_of_negative_probabilities = np.sum(negative_probabilities)
        normalized_negative_probabilities = np.array(negative_probabilities / sum_of_negative_probabilities)

        # get the uniform dist
        uniform_dist = np.ones(
            normalized_negative_probabilities.shape[0]) * 1 / normalized_negative_probabilities.shape[0]

        # calc the kl divergence
        kl_divergence = entropy(normalized_negative_probabilities, uniform_dist)
        flatten_target_words = [item for sublist in target_words_all_sets for item in sublist]

        negative_sentiment_distribution = list(zip(flatten_target_words, normalized_negative_probabilities))
        negative_sentiment_probabilities = list(zip(flatten_target_words, negative_probabilities))

        return kl_divergence, negative_sentiment_probabilities, negative_sentiment_distribution

    def run_query(self, query: Query, word_embedding: WordEmbeddingModel, classifier: BaseEstimator = None,
                  classifier_params: dict = None, print_model_evaluation: bool = False,
                  lost_vocabulary_threshold: float = 0.2, warn_filtered_words: bool = False) -> dict:
        """Calculates the RNSB metric over the provided parameters. 
        
        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute words for be tested.
        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contain certain word embedding pretrained model.
        classifier : BaseEstimator, optional
            A scikit-learn compatible classifier with predict_proba function, by default None,
        classifier_params : dict, optional
            The parameters that will use the the delivered classifier or the default classifier (logit), by default None
        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the training process is completed., by default False
        lost_vocabulary_threshold : bool, optional
            Indicates when a test is invalid due the loss of certain amount of words in any word set, by default 0.2
        warn_filtered_words : bool, optional
            A flag that indicates if the function will warn about the filtered words, by default False.
        
        Returns
        -------
        dict
            A dictionary with the kl-divergence, the negative probabilities for all tested target words 
            and the normalized distribution of probabilities.
        """

        # check the inputs
        self._check_input(query, word_embedding, lost_vocabulary_threshold, warn_filtered_words)

        # get the embeddings
        embeddings = self._get_embeddings_from_query(query, word_embedding, warn_filtered_words,
                                                     lost_vocabulary_threshold)
        # if there is any/some set has less words than the allowed limit, return the default value (nan)
        if embeddings is None:
            return {'query_name': query.query_name_, 'result': np.nan}

        # get the target and attribute embeddings dicts
        target_embeddings_dict, attribute_embeddings_dict = embeddings
        target_embeddings_all_sets = [list(target_dict.values()) for target_dict in target_embeddings_dict]
        target_words_all_sets = [list(target_dict.keys()) for target_dict in target_embeddings_dict]

        attribute_0_embeddings = list(attribute_embeddings_dict[0].values())
        attribute_1_embeddings = list(attribute_embeddings_dict[1].values())

        # train the logit with the train data.
        trained_classifier = self.__train_classifier(attribute_0_embeddings, attribute_1_embeddings, classifier,
                                                     classifier_params, print_model_evaluation)

        divergence, negative_sentiment_probabilities, negative_sentiment_distribution = self.__calc_rnsb(
            trained_classifier, target_embeddings_all_sets, target_words_all_sets)

        return {
            'query_name': query.query_name_,
            'result': divergence,
            'negative_sentiment_probabilities': negative_sentiment_probabilities,
            'negative_sentiment_distribution': negative_sentiment_distribution
        }
