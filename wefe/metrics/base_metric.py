"""Base metric class that all metrics must extend.."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Union

from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


class BaseMetric(ABC):
    """A base class to implement any metric following the framework described by WEFE.

    It contains the name of the metric, the templates (cardinalities) that it supports
    and the abstract function run_query, which must be implemented by any metric that
    extends this class.
    """

    # A tuple that indicates the cardinality of target and attribute sets
    metric_template: tuple[Union[int, str], Union[int, str]]

    # The name of the metric
    metric_name: str

    # The initials or short name of the metric
    metric_short_name: str

    def _check_input(
        self, query: Query, model: WordEmbeddingModel, _locals: dict[str, Any]
    ) -> None:
        """Check if Query and WordEmbeddingModel parameters are valid.

        Parameters
        ----------
        query : Query
            The query that the method will execute.
        model : WordEmbeddingModel
            A word embedding model.
        _locals: Dict[str, Any]
            The extra arguments of run_query.

        Raises
        ------
        TypeError
            if query is not instance of Query.
        TypeError
            if word_embedding is not instance of .
        TypeError
            if lost_vocabulary_threshold is not a float number.
        TypeError
            if warn_filtered_words is not a bool.
        Exception
            if the metric require different number of target sets than
            the delivered query
        Exception
            if the metric require different number of attribute sets than
            the delivered query

        """
        # check if the query passed is a instance of Query
        if not isinstance(query, Query):
            raise TypeError(f"query should be a Query instance, got {query}")

        # check if the word_embedding is a instance of
        if not isinstance(model, WordEmbeddingModel):
            raise TypeError(
                f"word_embedding should be a WordEmbeddingModel instance, got: {model}"
            )

        # templates:

        # check the cardinality of the target sets of the provided query
        if (
            self.metric_template[0] != "n"
            and query.template[0] != self.metric_template[0]
        ):
            raise Exception(
                f"The cardinality of the set of target words of the '{query.query_name}' "
                f"query does not match with the cardinality required by {self.metric_short_name}. "
                f"Provided query: {query.template[0]}, metric: {self.metric_template[0]}."
            )

        # check the cardinality of the attribute sets of the provided query
        if (
            self.metric_template[1] != "n"
            and query.template[1] != self.metric_template[1]
        ):
            raise Exception(
                "The cardinality of the set of attribute words of the "
                f"'{query.query_name}' query does not match with the cardinality "
                f"required by {self.metric_short_name}. Provided query: {query.template[1]}, metric: {self.metric_template[1]}."
            )

        preprocessor_in_args = "preprocessor_args" in _locals
        secondary_preprocessor_in_args = "secondary_preprocessor_args" in _locals

        if preprocessor_in_args and secondary_preprocessor_in_args:
            raise DeprecationWarning(
                "preprocessor_args and secondary_preprocessor_args arguments are "
                "deprecated. Use "
                f"preprocessors=[{_locals['preprocessor_args']}, "
                f"{_locals['secondary_preprocessor_args']}] "
                "instead.\n\nSee https://wefe.readthedocs.io/en/latest/user_guide_"
                "measurement.html#word-preprocessors for more information."
            )
        if preprocessor_in_args:
            raise DeprecationWarning(
                "preprocessor_args argument is deprecated. Use "
                f"preprocessors=[{_locals['preprocessor_args']}] "
                "instead.\n\nSee https://wefe.readthedocs.io/en/latest/user_guide_"
                "measurement.html#word-preprocessors for more information."
            )
        if secondary_preprocessor_in_args:
            raise DeprecationWarning(
                "secondary_preprocessor_args is deprecated. Use "
                f"preprocessors=[{{}}, {_locals['secondary_preprocessor_args']}] "
                "instead.\n\nSee https://wefe.readthedocs.io/en/latest/user_guide_"
                "measurement.html#word-preprocessors for more information."
            )

    @abstractmethod
    def run_query(
        self,
        query: Query,
        model: WordEmbeddingModel,
        lost_vocabulary_threshold: float = 0.2,
        preprocessors: list[dict[str, Union[str, bool, Callable]]] = [{}],
        strategy: str = "first",
        normalize: bool = False,
        warn_not_found_words: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError()
