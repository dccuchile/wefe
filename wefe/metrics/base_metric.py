"""Base metric class that all metrics must extend.."""

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Union

from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class BaseMetric(ABC):
    """An abstract base class for implementing fairness metrics in the WEFE.

    This class provides a template for all metrics, ensuring consistent input
    validation and parameter handling. Subclasses are required to define metric-specific
    attributes and implement the core calculation logic in the `run_query`
    method.

    Attributes
    ----------
    metric_template : ClassVar[tuple[Union[int, str], Union[int, str]]]
        A tuple indicating the required cardinality of target and attribute sets,
        e.g., (1, 1) or (2, 'n'). 'n' denotes any number of sets.
        This must be overridden by subclasses.

    metric_name : ClassVar[str]
        The full name of the metric.
        This must be overridden by subclasses.

    metric_short_name : ClassVar[str]
        The initials or a short name for the metric.
        This must be overridden by subclasses.

    """

    # These attributes MUST be overridden by any class that extends BaseMetric.
    metric_template: ClassVar[tuple[Union[int, str], Union[int, str]]]
    metric_name: ClassVar[str]
    metric_short_name: ClassVar[str]

    def _check_input(
        self,
        query: Query,
        model: WordEmbeddingModel,
        lost_vocabulary_threshold: float,
        warn_not_found_words: bool,
    ) -> None:
        """Check if the parameters for run_query are valid.

        This private method is called by the `run_query` template method.

        Parameters
        ----------
        query : Query
            The query that the method will execute.
        model : WordEmbeddingModel
            A word embedding model.
        lost_vocabulary_threshold : float
            The threshold for the proportion of lost words.
        warn_not_found_words : bool
            Specifies whether to warn about out-of-vocabulary words.


        Raises
        ------
        TypeError
            If `query` is not an instance of `Query`.
        TypeError
            If `model` is not an instance of `WordEmbeddingModel`.
        TypeError
            If `lost_vocabulary_threshold` is not a float.
        TypeError
            If `warn_not_found_words` is not a bool.
        ValueError
            If the query's template cardinality does not match the metric's
            required template.

        """
        if not isinstance(query, Query):
            raise TypeError(f"query should be a Query instance, got: {type(query)}.")
        if not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                f"model should be a WordEmbeddingModel instance, got: {type(model)}."
            )

        if not isinstance(lost_vocabulary_threshold, float):
            raise TypeError(
                f"lost_vocabulary_threshold should be a float, got: "
                f"{type(lost_vocabulary_threshold)}."
            )

        if not isinstance(warn_not_found_words, bool):
            raise TypeError(
                f"warn_not_found_words should be a bool, got: "
                f"{type(warn_not_found_words)}."
            )

        # Check the cardinality of the target sets
        if (
            self.metric_template[0] != "n"
            and query.template[0] != self.metric_template[0]
        ):
            raise ValueError(
                f"The cardinality of the target sets of the '{query.query_name}' "
                f"query ({query.template[0]}) does not match the cardinality "
                f"required by {self.metric_short_name} ({self.metric_template[0]})."
            )

        # Check the cardinality of the attribute sets
        if (
            self.metric_template[1] != "n"
            and query.template[1] != self.metric_template[1]
        ):
            raise ValueError(
                f"The cardinality of the attribute sets of the '{query.query_name}' "
                f"query ({query.template[1]}) does not match the cardinality "
                f"required by {self.metric_short_name} ({self.metric_template[1]})."
            )

    @abstractmethod
    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: list[dict[str, str | bool | Callable]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Runs the metric on the given query and model.

        Parameters
        ----------
        query : Query
            A Query object that contains the target and attribute word sets to
            be tested.
        model : WordEmbeddingModel
            A word embedding model.
        lost_vocabulary_threshold : float, optional
            Specifies the proportional limit of words that any set of the query is
            allowed to lose when transforming its words into embeddings.
            In the case that any set of the query loses proportionally more words
            than this limit, the result values will be np.nan, by default 0.2.
        preprocessors : list[dict[str, str | bool | Callable]]
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
        dict[str, Any]
            A dictionary containing the results of the metric.

        """
        raise NotImplementedError()
