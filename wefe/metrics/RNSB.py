import logging
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from scipy.stats import entropy

from ..utils import verify_metric_input, get_embeddings_from_word_set, verify_vocabulary_threshold
from ..query import Query
from ..word_embedding_model import WordEmbeddingModel


class RNSB:
    """Relative Negative Sentiment Bias
    """

    def __init__(self):
        self.template_required = ('n', 2)
        self.method_name = 'Relative Negative Sentiment Bias (RNSB)'
        self.abbreviated_method_name = 'RNSB'

    def __train_classifier(self, labeled_attributes_embeddings: list, attributes_labels: list, Classifier,
                           classifier_params: dict, random_state: int, evaluate_model: bool,
                           disable_vocab_warnings: bool):

        # split the filtered words in train and test sets.
        X_embeddings_train, X_embeddings_test, y_train, y_test = train_test_split(labeled_attributes_embeddings,
                                                                                  attributes_labels, test_size=0.33)
        if classifier_params is None:
            if not (random_state is None):
                classifier_params = {'random_state': random_state}
            else:
                classifier_params = {}
        else:
            if not (random_state is None) and 'random_state' not in classifier_params:
                classifier_params['random_state'] = random_state

        # train logistic classifier
        if not (Classifier is None):
            classifier = Classifier(**classifier_params)

        else:
            classifier_params['solver'] = 'lbfgs'
            classifier_params['max_iter'] = 2000
            classifier_params['multi_class'] = 'multinomial'
            classifier = LogisticRegression(**classifier_params)

        classifier.fit(X_embeddings_train, y_train)

        # evaluate
        if evaluate_model:
            y_pred = classifier.predict(X_embeddings_test)
            print(classifier)
            print("Classification Report\n", classification_report(y_test, y_pred, labels=classifier.classes_))
            print('\n------------------------------------------------------------')

        return classifier

    def run_query(self, query: Query, word_vectors_wrapper: WordEmbeddingModel, lost_vocabulary_threshold: float = 0.2,
                  average='micro', random_state: int = None, Classifier=None, classifier_params: dict = None,
                  evaluate_model: bool = False, disable_vocab_warnings: bool = True):

        # check the inputs
        verify_metric_input(query, word_vectors_wrapper, self.template_required, self.method_name)

        if average not in ['macro', 'micro']:
            raise Exception('Average must be \'macro\' or \'micro\'')

        target_sets = query.target_sets
        target_sets_names = query.target_sets_names

        attribute_1_words = query.attribute_sets[0]
        attribute_2_words = query.attribute_sets[1]
        attribute_1_name = query.attribute_sets_names[0]
        attribute_2_name = query.attribute_sets_names[1]

        default = {
            'exp_name': ', '.join(target_sets_names) + ' wrt {} and {}'.format(attribute_1_name, attribute_2_name),
            'result': np.nan
        }

        word_embeddings_name = word_vectors_wrapper.model_name

        # transform and check all target set inputs
        target_sets_embeddings = []
        for target_set_idx, target_set_words in enumerate(target_sets):

            target_set_embeddings, target_set_lost_words = get_embeddings_from_word_set(
                target_set_words, word_vectors_wrapper, warn_filtered_words=disable_vocab_warnings)

            if not verify_vocabulary_threshold(target_set_words, target_set_lost_words, lost_vocabulary_threshold,
                                               target_sets_names[target_set_idx], word_embeddings_name):
                return default
            else:
                target_sets_embeddings.append(target_set_embeddings)
        target_sets_embeddings = np.array(target_sets_embeddings)
        # transform and check all attribute input
        attribute_1_embeddings, attribute_1_lost_words = get_embeddings_from_word_set(
            attribute_1_words, word_vectors_wrapper, warn_filtered_words=disable_vocab_warnings)
        attribute_2_embeddings, attribute_2_lost_words = get_embeddings_from_word_set(
            attribute_2_words, word_vectors_wrapper, warn_filtered_words=disable_vocab_warnings)
        if not verify_vocabulary_threshold(attribute_1_words, attribute_1_lost_words, lost_vocabulary_threshold,
                                           attribute_1_name, word_embeddings_name):
            return default

        if not verify_vocabulary_threshold(attribute_2_words, attribute_2_lost_words, lost_vocabulary_threshold,
                                           attribute_2_name, word_embeddings_name):
            return default

        # label the attribute set
        negative_attribute_labels = [-1 for embedding in attribute_1_embeddings]
        positive_attribute_labels = [1 for embedding in attribute_2_embeddings]

        attributes_embeddings = np.concatenate((attribute_1_embeddings, attribute_2_embeddings))
        attributes_labels = negative_attribute_labels + positive_attribute_labels

        # train the logit with the train data.
        logit = self.__train_classifier(attributes_embeddings, attributes_labels, Classifier, classifier_params,
                                        random_state, evaluate_model, disable_vocab_warnings)

        if (average == 'macro'):
            # get the probabilities associated with each target word vector
            probabilities = np.array(
                [logit.predict_proba(target_set_word_vectors) for target_set_word_vectors in target_sets_embeddings])

            # extract only the negative sentiment probability for each word
            negative_probabilities = np.array([probability[:, 0] for probability in probabilities])

            # average the negative sentiment for each word.reformulated for target sets with different shapes.
            average_negative_probabilities = []
            for negative_probability_array in negative_probabilities:
                # calc the average of the negative sentiment probability
                average_negative_probabilities.append(negative_probability_array.mean())

            # normalization of the probabilities
            sum_of_negative_probabilities = np.sum(average_negative_probabilities)
            normalized_negative_probabilities = np.array(average_negative_probabilities / sum_of_negative_probabilities)

            # for the divergence, count the number of target words
            number_of_target_words = target_sets_embeddings.shape[0]

            # get the uniform dist
            uniform_dist = np.ones(probabilities.shape[0]) * 1 / number_of_target_words

            # calc the kl divergence
            kl_divergence = entropy(normalized_negative_probabilities, uniform_dist)

            return {
                'exp_name': ', '.join(target_sets_names) + ' wrt {} and {}'.format(attribute_1_name, attribute_2_name),
                'result': kl_divergence,
                'negative_sentiment_probabilities': [
                    list(zip(target_set_probabilities[0], target_set_probabilities[1]))
                    for target_set_probabilities in zip(target_sets, negative_probabilities)
                ],
                'negative_sentiment_distribution': list(zip(target_sets_names, normalized_negative_probabilities))
            }

        # calculate RNSB averaging the negative sentiment of each word of each target set.
        elif average == 'micro':

            # get the probabilities associated with each target word vector
            probabilities = np.array(
                [logit.predict_proba(target_set_word_vectors) for target_set_word_vectors in target_sets_embeddings])

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

            return {
                'exp_name': ', '.join(target_sets_names) + ' wrt {} and {}'.format(attribute_1_name, attribute_2_name),
                'result': kl_divergence,
                'negative_sentiment_probabilities': negative_probabilities,
                'negative_sentiment_distribution': list(zip(target_sets_names, normalized_negative_probabilities))
            }
