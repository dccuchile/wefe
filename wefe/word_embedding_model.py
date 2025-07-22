"""A Word Embedding contanier based on gensim BaseKeyedVectors."""

from collections.abc import Sequence
from typing import Any

import gensim
import numpy as np
import semantic_version
from numpy.typing import NDArray

GENSIM_VERSION = semantic_version.Version.coerce(gensim.__version__)
GENSIM_V4_OR_GREATER = GENSIM_VERSION.major >= 4  # type: ignore

if GENSIM_V4_OR_GREATER:
    from gensim.models import KeyedVectors as BaseKeyedVectors
else:
    # In older versions, BaseKeyedVectors is in a different location.
    from gensim.models.keyedvectors import BaseKeyedVectors  # type: ignore

# --- Type Aliases ---
# Using NDArray for better type hinting with NumPy arrays.
EmbeddingDict = dict[str, NDArray[np.float64]]
EmbeddingSets = dict[str, EmbeddingDict]


class WordEmbeddingModel:
    """A wrapper for Word Embedding pre-trained models.

    It can hold gensim's KeyedVectors or gensim's api loaded models.
    It includes the name of the model and some vocab prefix if needed.
    """

    def __init__(
        self,
        wv: BaseKeyedVectors,
        name: str | None = None,
        vocab_prefix: str | None = None,
    ) -> None:
        """Initialize the word embedding model.

        Parameters
        ----------
        wv : BaseKeyedVectors.
            An instance of word embedding loaded through gensim KeyedVector
            interface or gensim's api.
        name : str, optional
            The name of the model, by default ''.
        vocab_prefix : str, optional.
            A prefix that will be concatenated with all word in the model
            vocab, by default None.

        Raises
        ------
        TypeError
            if word_embedding is not a KeyedVectors instance.
        TypeError
            if model_name is not None and not an instance of str.
        TypeError
            if vocab_prefix is not None and not an instance of str.


        Examples
        --------
        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from wefe.word_embedding_model import WordEmbeddingModel

        >>> dummy_model = Word2Vec(common_texts, window=5,
        ...                        min_count=1, workers=1).wv

        >>> model = WordEmbeddingModel(dummy_model, 'Dummy model dim=10',
        ...                            vocab_prefix='/en/')
        >>> print(model.name)
        Dummy model dim=10
        >>> print(model.vocab_prefix)
        /en/

        """
        # Type checking
        if not isinstance(wv, BaseKeyedVectors):
            raise TypeError(
                f"wv must be an instance of gensim's BaseKeyedVectors, "
                f"but got {type(wv)}."
            )
        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, but got {type(name)}.")
        if vocab_prefix is not None and not isinstance(vocab_prefix, str):
            raise TypeError(
                f"vocab_prefix must be a string or None, but got {type(vocab_prefix)}."
            )

        # Assign the attributes
        self.wv = wv

        if GENSIM_V4_OR_GREATER:
            self.vocab = self.wv.key_to_index
        else:
            self.vocab = self.wv.vocab

        self.vocab_prefix = vocab_prefix
        if name is None:
            self.name = "Unnamed model"
        else:
            self.name = name

    def __eq__(self, other: Any) -> bool:
        """Check if other is the same WordEmbeddingModel that self.

        Parameters
        ----------
        other : Any
            Some object

        Returns
        -------
        bool
            True if other is a WordEmbeddingModel that have the same model, model_name
            and vocab_prefix . False in any other case

        """
        if not isinstance(other, WordEmbeddingModel):
            return False
        if self.wv != other.wv:
            return False
        if self.name != other.name:
            return False
        return self.vocab_prefix == other.vocab_prefix

    def __len__(self) -> int:
        """Return the number of words in the vocabulary."""
        return len(self.vocab)

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        """Retrieve the embedding for a word.

        Parameters
        ----------
        key : str
            A word

        Returns
        -------
        np.ndarray
            The embedding associated with the word.

        Raises
        ------
        KeyError
            If the word is not in the vocabulary.

        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, but got {type(key)}.")

        if key not in self.vocab:
            raise KeyError(f"word '{key}' not in model vocab.")

        return self.wv[key]

    def __contains__(self, key: str) -> bool:
        """Check if a word is in the model's vocabulary.

        Parameters
        ----------
        key: str
            Some word.

        Returns
        -------
        bool
            True if the word exists in the model's vocabulary.

        """
        return key in self.vocab

    def __repr__(self) -> str:
        """Generate a string representation of the WordEmbeddingModel.

        Format:

        <WordEmbeddingModel named {name} with {n_embeddings} of {dims} dims>

        Returns
        -------
        str
            The generated representation.

        """
        try:
            if self.name == "Unnamed model" and self.vocab_prefix is not None:
                return (
                    "<WordEmbeddingModel 'Unnamed model' "
                    f"with {self.wv.vectors.shape[0]}"
                    f" word embeddings of {self.wv.vectors.shape[1]} dims"
                    f" and '{self.vocab_prefix}' as word prefix>"
                )

            if self.name == "Unnamed model":
                return (
                    "<WordEmbeddingModel 'Unnamed model' "
                    f"with {self.wv.vectors.shape[0]}"
                    f" word embeddings of {self.wv.vectors.shape[1]} dims>"
                )

            if self.vocab_prefix is not None:
                return (
                    f"<WordEmbeddingModel named '{self.name}' with "
                    f"{self.wv.vectors.shape[0]}"
                    f" word embeddings of {self.wv.vectors.shape[1]} dims"
                    f" and '{self.vocab_prefix}' as word prefix>"
                )

            return (
                f"<WordEmbeddingModel named '{self.name}' "
                f"with {self.wv.vectors.shape[0]}"
                f" word embeddings of {self.wv.vectors.shape[1]} dims>"
            )
        except AttributeError:
            # it can happen if some of the attributes (name or vocab_prefix) are not
            # defined.
            return "<WordEmbeddingModel with wrong __repr__>"

    def get(self, word: str, default: Any | None = None) -> NDArray[np.float64] | None:
        """Retrieve a word's embedding, returning a default value if not found."""
        return self.wv[word] if word in self else default

    def normalize(self) -> None:
        """Normalize the word vectors to unit L2 length.

        This method uses the underlying gensim functionality to perform
        L2 normalization. The model's vectors are modified in-place.
        **Warning**: This is a destructive operation.


        Raises
        ------
        AttributeError
            If the underlying model does not support normalization.

        """
        # Gensim 4+ has a more direct way to get normalized vectors.
        # To maintain the "inplace" behavior, we re-assign the vectors.
        if hasattr(self.wv, "get_normed_vectors"):
            self.wv.vectors = self.wv.get_normed_vectors()
            # Ensure the norms are also updated for similarity calculations
            if GENSIM_V4_OR_GREATER:
                self.wv.fill_norms(force=True)
        elif hasattr(self.wv, "init_sims"):
            self.wv.init_sims(replace=True)
        else:
            raise AttributeError(
                "The underlying gensim model does not have a "
                "known normalization method ('get_normed_vectors' or 'init_sims')."
            )

    def update(self, word: str, embedding: NDArray[np.float64]) -> None:
        """Update the value of an embedding of the model.

        If the method is executed with a word that is not in the vocabulary, an
        exception will be raised.

        Parameters
        ----------
        word : str
            The word to update. It must already exist in the vocabulary.
        embedding : NDArray[np.float64]
            The new embedding for the word. Must match the model's vector size
            and dtype.

        Raises
        ------
        TypeError
            if word is not a1 string.
        TypeError
            if embedding is not an np.array.
        ValueError
            if word is not in the model's vocabulary.
        ValueError
            if the embedding is not the same size as the size of the model's embeddings.
        ValueError
            if the dtype of the embedding values is not the same as the model's
            embeddings.

        """
        if not isinstance(word, str):
            raise TypeError(f"Word must be a string, but got {type(word)}.")
        if word not in self:
            raise ValueError(f"Word '{word}' is not in the model's vocabulary.")
        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                f"Embedding must be a NumPy array, but got {type(embedding)}."
            )
        embedding_size = embedding.shape[0]
        if self.wv.vector_size != embedding_size:
            raise ValueError(
                f"The size of '{word}' embedding ({embedding_size}) is different from "
                f"the size of the embeddings in the model ({self.wv.vector_size})."
            )

        if not np.issubdtype(self.wv.vectors.dtype, embedding.dtype):
            raise ValueError(
                f"embedding dtype ({embedding.dtype}) is not the same of model's dtype "
                f"({self.wv.vectors.dtype})."
            )

        if GENSIM_V4_OR_GREATER:
            idx = self.wv.key_to_index[word]
        else:
            idx = self.wv.vocab[word].index
        self.wv.vectors[idx] = embedding.astype(self.wv.vectors.dtype)

    def batch_update(
        self,
        words: Sequence[str],
        embeddings: Sequence[np.ndarray] | np.ndarray,
    ) -> None:
        """Update a batch of embeddings in the model.

        This method updates the embeddings for a given sequence of words efficiently
        by leveraging NumPy's advanced indexing. All validations (word existence,
        embedding shape, and data type) are performed collectively before any
        modifications are applied to the model. This ensures atomicity: either all
        updates succeed, or none do.

        Parameters
        ----------
        words : Sequence[str]
            A sequence (list, tuple, or np.ndarray) containing the words whose
            representations will be updated. All words must already exist in
            the model's vocabulary and must be strings.
        embeddings : Union[Sequence[np.ndarray], np.ndarray]
            A sequence (list or tuple) of NumPy arrays, or a 2D NumPy array, that
            contains all the new embeddings. Each embedding must be a 1D NumPy
            array with the same size and data type as the model's embeddings.
            The length of `embeddings` must match the length of `words`.

        Raises
        ------
        TypeError
            If `words` is not a sequence of strings, or if `embeddings` is not
            a sequence of NumPy arrays or a single NumPy array.
            Also, if individual elements within `words` are not strings, or elements
            within `embeddings` are not NumPy arrays.
        ValueError
            If `words` and `embeddings` do not have the same number of elements.
            If any word in `words` is not found in the model's vocabulary.
            If any embedding has a different dimension than the model's embeddings.
            If any embedding has a data type incompatible with the model's embeddings.

        Examples
        --------
        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from wefe.word_embedding_model import WordEmbeddingModel
        >>> import numpy as np

        >>> # Create a dummy WordEmbeddingModel
        >>> kv_model = Word2Vec(common_texts, vector_size=10, min_count=1).wv
        >>> model = WordEmbeddingModel(kv_model, 'Dummy Model')
        >>> original_embedding_the = model['the']
        >>> original_embedding_system = model['system']

        >>> # Prepare words and new embeddings
        >>> words_to_update = ['the', 'system']
        >>> new_embeddings = [
        ...     np.zeros(10, dtype=model.wv.vectors.dtype),
        ...     np.ones(10, dtype=model.wv.vectors.dtype)
        ... ]

        >>> # Update embeddings
        >>> model.batch_update(words_to_update, new_embeddings)

        >>> # Verify updates
        >>> assert np.all(model['the'] == np.zeros(10))
        >>> assert np.all(model['system'] == np.ones(10))

        >>> # Test with missing word (will raise error)
        >>> try:
        ...     model.batch_update(['nonexistent_word'], [np.zeros(10)])
        ... except ValueError as e:
        ...     print(e)
        The following words are not in the model's vocabulary: nonexistent_word.

        """
        # Initial type and length validation for the input containers
        if not isinstance(words, (list, tuple, np.ndarray)):
            raise TypeError(
                f"words argument should be a list, tuple or np.array of strings, "
                f"but got {type(words)}."
            )
        if not isinstance(embeddings, (list, tuple, np.ndarray)):
            raise TypeError(
                "embeddings argument should be a list, tuple or np.array of "
                f"NumPy arrays, but got {type(embeddings)}."
            )
        if len(words) != len(embeddings):
            raise ValueError(
                "words and embeddings must have the same number of elements, "
                f"but got {len(words)} words and {len(embeddings)} embeddings."
            )

        # 1. Validate 'words' elements and collect their indices
        missing_words = []
        word_indices = []
        for word in words:
            if not isinstance(word, str):
                raise TypeError(
                    f"All elements in 'words' must be strings, but found a "
                    f"{type(word)}."
                )
            if word not in self.vocab:
                missing_words.append(word)
            else:
                # Get the index based on gensim version
                if GENSIM_V4_OR_GREATER:
                    word_indices.append(self.wv.key_to_index[word])
                else:
                    word_indices.append(self.wv.vocab[word].index)

        if missing_words:
            raise ValueError(
                f"The following words are not in the model's vocabulary: "
                f"{', '.join(missing_words)}."
            )

        # Convert collected indices to a NumPy array for advanced indexing
        np_word_indices = np.array(word_indices, dtype=int)

        # Define expected properties for the embeddings based on the model
        expected_vector_size = self.wv.vector_size
        model_dtype = self.wv.vectors.dtype

        # 2. Validate and prepare 'embeddings' for batch update
        embeddings_to_update: np.ndarray

        # If 'embeddings' is already a 2D NumPy array, perform checks directly on it.
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim != 2 or embeddings.shape[1] != expected_vector_size:
                raise ValueError(
                    f"Input embeddings array has shape {embeddings.shape}, "
                    f"but expected a 2D array with {expected_vector_size} columns "
                    f"(model's vector size {expected_vector_size})."
                )
            if not np.can_cast(
                embeddings.dtype, model_dtype
            ):  # Check if source dtype can be cast to target dtype
                raise ValueError(
                    f"Input embeddings array dtype ({embeddings.dtype}) cannot be "
                    f"safely cast to model's dtype ({model_dtype})."
                )
            # Ensure correct dtype for assignment; copy=False avoids copy if
            # already correct
            embeddings_to_update = embeddings.astype(model_dtype, copy=False)
        else:  # Handle Sequence of np.ndarray
            temp_embeddings_list = []
            for i, emb in enumerate(embeddings):
                # Ensure each element is a NumPy array
                if not isinstance(emb, np.ndarray):
                    raise TypeError(
                        f"Embedding at index {i} ('{words[i]}') is not a NumPy array, "
                        f"but got {type(emb)}."
                    )
                # Ensure each embedding has the correct dimension (1D) and size
                if emb.ndim != 1 or emb.shape[0] != expected_vector_size:
                    raise ValueError(
                        f"Embedding at index {i} ('{words[i]}') has shape {emb.shape} "
                        f"which is different from the model's embedding size "
                        f"({expected_vector_size},)."
                    )
                # Ensure data type compatibility
                if not np.can_cast(
                    emb.dtype, model_dtype
                ):  # Check if source dtype can be cast to target dtype
                    raise ValueError(
                        f"Embedding at index {i} ('{words[i]}') with dtype "
                        f"({emb.dtype}) "
                        f"cannot be safely cast to model's dtype ({model_dtype})."
                    )
                temp_embeddings_list.append(emb)  # Collect validated embeddings

            # Convert the list of validated embeddings to a single 2D NumPy array
            embeddings_to_update = np.array(temp_embeddings_list, dtype=model_dtype)

        # 3. Perform the batch update using advanced indexing
        self.wv.vectors[np_word_indices] = embeddings_to_update
