from typing import Any, Dict, Union
import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from .base_metric import BaseMetric
from ..query import Query
from ..word_embedding_model import PreprocessorArgs, WordEmbeddingModel


class ECT(BaseMetric):
    """An implementation of the Embedding Coherence Test.

    The metrics was originally proposed in [1] and implemented in [2].

    The general steps of the test, as defined in [1], are as follows:

    1. Embedd all given target and attribute words with the given embedding model
    2. Calculate mean vectors for the two sets of target word vectors
    3. Measure the cosine similarity of the mean target vectors to all of the given attribute words
    4. Calculate the Spearman r correlation between the resulting two lists of similarities
    5. Return the correlation value as score of the metric (in the range of -1 to 1); higher is
       better

    References
    ----------

    | [1]: Dev, S., & Phillips, J. (2019, April). Attenuating Bias in Word vectors.
    | [2]: https://github.com/sunipa/Attenuating-Bias-in-Word-Vec

    """

    # The metrics accepts two target sets and a single attribute set
    metric_template = (2, 1)
    metric_name = 'Embedding Coherence Test'
    metric_short_name = 'ECT'

    def run_query(self,
                  query: Query,
                  word_embedding: WordEmbeddingModel,
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
        """Runs ECT with the given query with the given parameters.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to 
            be tested.

        word_embedding : 
            A  object that contains certain word embedding 
            pretrained model.

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
            A dictionary with the query name and the result of the query.
        """

        # Standard input procedure: check the inputs and obtain the
        # embeddings.
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

        # If the lost vocabulary threshold is exceeded, return the default value
        if embeddings is None:
            return {"query_name": query.query_name, "result": np.nan}

        # get the targets and attribute sets transformed into embeddings.
        target_sets, attribute_sets = embeddings

        # get only the embeddings of the sets.
        target_embeddings = list(target_sets.values())
        attribute_embeddings = list(attribute_sets.values())

        ect = self.__calculate_embedding_coherence(
            list(target_embeddings[0].values()), list(target_embeddings[1].values()),
            list(attribute_embeddings[0].values()))

        return {"query_name": query.query_name, "result": ect, 'ect': ect}

    def __calculate_embedding_coherence(self, target_set_1: list, target_set_2: list,
                                        attribute_set: list) -> float:
        """Calculate the ECT metric over the given parameters. Return the result.

        Parameters
        ----------
        target_set_1 : list
            The first set of target words.
        target_set_2 : list
            The second set of target words.
        attribute_set : list
            The set of attribute words.

        Returns
        -------
        float
            The value denoting the Spearman correlation.
        """

        # Calculate mean vectors for both target vector sets
        target_means = [np.mean(s, axis=0) for s in (target_set_1, target_set_2)]

        # Measure similarities between mean vecotrs and all attribute words
        similarities = []
        for mean_vector in target_means:
            similarities.append([1 - cosine(mean_vector, a) for a in attribute_set])

        # Calculate similarity correlations
        return spearmanr(similarities[0], similarities[1]).correlation
