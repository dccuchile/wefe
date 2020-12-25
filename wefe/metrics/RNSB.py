from typing import Any, Dict, Tuple, List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
from scipy.stats import entropy
import logging

from ..query import Query
from ..word_embedding_model import PreprocessorArgs, WordEmbeddingModel
from .base_metric import BaseMetric

logging.basicConfig(level=logging.DEBUG)


class RNSB(BaseMetric):
    """A implementation of Relative Relative Negative Sentiment Bias (RNSB).

    References
    ----------
    [1] Chris Sweeney and Maryam Najafian.
        A transparent framework for evaluating unintended demographic bias in word
        embeddings.
        In Proceedings of the 57th Annual Meeting of the Association for
        Computational Linguistics, pages 1662–1667, 2019.
    """

    metric_template = ('n', 2)
    metric_name = 'Relative Negative Sentiment Bias'
    metric_short_name = 'RNSB'

    def _train_classifier(
            self,
            attribute_embeddings_dict: List[Dict[str, np.ndarray]],
            estimator: BaseEstimator = LogisticRegression,
            estimator_params: Dict[str, Any] = {
                'solver': 'liblinear',
                'max_iter': 10000,
            },
            random_state: Union[int, None] = None,
            print_model_evaluation: bool = False) -> Tuple[BaseEstimator, float]:
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
            Seed that allows to make the execution of the query reproducible. 
            by default None

        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed., by default False

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
            (attribute_0_embeddings, attribute_1_embeddings))
        attributes_labels = np.concatenate(
            (negative_attribute_labels, positive_attribute_labels))

        split = train_test_split(attributes_embeddings,
                                 attributes_labels,
                                 shuffle=True,
                                 random_state=random_state,
                                 test_size=0.33,
                                 stratify=attributes_labels)
        X_embeddings_train, X_embeddings_test, y_train, y_test = split

        num_train_negative_examples = np.count_nonzero((y_train == -1))
        num_train_positive_examples = np.count_nonzero((y_train == 1))

        # Check the number of train and test examples.
        if num_train_positive_examples == 1:
            raise Exception(
                'After dividing the datset using stratified train_test_split, '
                'the attribute 0 has 0 training examples.')

        if num_train_negative_examples < 1:
            raise Exception(
                'After dividing the datset using stratified train_test_split, '
                'the attribute 1 has 0 training examples.')

        # when random_state is not none, set it on classifier params.
        if random_state is not None:
            estimator_params['random_state'] = random_state

        estimator = estimator(**estimator_params)
        estimator.fit(X_embeddings_train, y_train)

        # evaluate
        y_pred = estimator.predict(X_embeddings_test)
        score = estimator.score(X_embeddings_test, y_test)

        if print_model_evaluation:
            print("Classification Report:\n{}".format(
                classification_report(y_test, y_pred, labels=estimator.classes_)))

        return estimator, score

    def _calc_rnsb(self, target_embeddings_dict: List[Dict[str, np.ndarray]],
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
            list(target_dict.values()) for target_dict in target_embeddings_dict
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
        normalized_negative_probabilities = np.array(negative_probabilities /
                                                     sum_of_negative_probabilities)

        # get the uniform dist
        uniform_dist = np.ones(normalized_negative_probabilities.shape[0]
                               ) * 1 / normalized_negative_probabilities.shape[0]

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

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
                  estimator: BaseEstimator = LogisticRegression,
                  estimator_params: Dict[str, Any] = {
                      'solver': 'liblinear',
                      'max_iter': 10000,
                  },
                  num_iterations: int = 1,
                  random_state: Union[int, None] = None,
                  print_model_evaluation: bool = False,
                  lost_vocabulary_threshold: float = 0.2,
                  preprocessor_args: PreprocessorArgs = {
                      'strip_accents': False,
                      'lowercase': False,
                      'preprocessor': None,
                  },
                  secondary_preprocessor_args: PreprocessorArgs = None,
                  warn_not_found_words: bool = False,
                  *args: Any,
                  **kwargs: Any) -> Dict[str, Any]:
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

        word_embedding : WordEmbeddingModel
            A WordEmbeddingModel object that contains certain word embedding 
            pretrained model.

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
            Seed that allows to make the execution of the query reproducible. 
            Warning: if a random_state other than None is provided along with 
            num_iterations, each iteration will split the dataset and train a 
            classifier associated to the same seed, so the results of each iteration 
            will always be the same , by default None.

        print_model_evaluation : bool, optional
            Indicates whether the classifier evaluation is printed after the
            training process is completed., by default False

        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is 
            allowed to lose when transforming its words into embeddings. 
            In the case that any set of the query loses proportionally more words 
            than this limit, the result values will be np.nan, by default 0.2
        
        preprocessor_args : PreprocessorArgs, optional
            Dictionary with the arguments that specify how the pre-processing of the 
            words will be done, by default {}
            The possible arguments for the function are: 
            - lowercase: bool. Indicates if the words are transformed to lowercase.
            - strip_accents: bool, {'ascii', 'unicode'}: Specifies if the accents of 
                             the words are eliminated. The stripping type can be 
                             specified. True uses 'unicode' by default.
            - preprocessor: Callable. It receives a function that operates on each 
                            word. In the case of specifying a function, it overrides 
                            the default preprocessor (i.e., the previous options 
                            stop working).
            , by default { 'strip_accents': False, 'lowercase': False, 'preprocessor': None, }
        
        secondary_preprocessor_args : PreprocessorArgs, optional
            Dictionary with the arguments that specify how the secondary pre-processing 
            of the words will be done, by default None.
            Indicates that in case a word is not found in the model's vocabulary 
            (using the default preprocessor or specified in preprocessor_args), 
            the function performs a second search for that word using the preprocessor 
            specified in this parameter.

        warn_not_found_words : bool, optional
            Specifies if the function will warn (in the logger)
            the words that were not found in the model's vocabulary
            , by default False.


        Returns
        -------
        Dict[str, Any]
            A dictionary with the query name, the calculated kl-divergence, 
            the negative probabilities for all tested target words and 
            the normalized distribution of probabilities.
        """
        # checks the types of the provided arguments (only the defaults).
        super().run_query(query, word_embedding, lost_vocabulary_threshold,
                          preprocessor_args, secondary_preprocessor_args,
                          warn_not_found_words, *args, **kwargs)

        # transforming query words into embeddings
        embeddings = word_embedding.get_embeddings_from_query(
            query=query,
            lost_vocabulary_threshold=lost_vocabulary_threshold,
            preprocessor_args=preprocessor_args,
            secondary_preprocessor_args=secondary_preprocessor_args,
            warn_not_found_words=warn_not_found_words)

        # if there is any/some set has less words than the allowed limit,
        # return the default value (nan)
        if embeddings is None:
            return {
                'query_name': query.query_name,
                'result': np.nan,
                'kl-divergence': np.nan,
                'score': np.nan,
                'negative_sentiment_probabilities': {},
                'negative_sentiment_distribution': {}
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
                print_model_evaluation=print_model_evaluation)

            scores.append(score)

            # get the scores
            divergence, negative_sentiment_probabilities = self._calc_rnsb(
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
            'query_name': query.query_name,
            'result': divergence,
            'kl-divergence': divergence,
            'clf_accuracy': np.mean(scores),
            'negative_sentiment_probabilities': negative_sentiment_probabilities,
            'negative_sentiment_distribution': negative_sentiment_distribution
        }
