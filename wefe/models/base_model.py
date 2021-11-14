"""Base clase for Word Embedding contaniers based on multiple sources."""
from typing import Callable, Dict, Sequence, Union

import numpy as np
import semantic_version

EmbeddingDict = Dict[str, np.ndarray]
EmbeddingSets = Dict[str, EmbeddingDict]

class BaseModel:
    def __init__(self, name: str = None, vocab_prefix: str = None):
        if not isinstance(name, (str, type(None))):
            raise TypeError("name should be a string or None, got {}.".format(name))

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError("vocab_prefix should be a string or None, got {}".format(vocab_prefix))
        
        if name is None:
            self.name = "Unnamed model"
        else:
            self.name = name
        
        self.wv           = None
        self.vocab        = None
        self.context      = False
        self.vocab_prefix = vocab_prefix
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseModel):
            return False
        if self.wv != other.wv:
            return False
        if self.name != other.name:
            return False
        if self.vocab_prefix != other.vocab_prefix:
            return False
        return True
    
    def __getitem__(self, key: str) -> Union[np.ndarray, None]:
        return None

    def __getitem__(self, target: str, attribute: str, template: str):
        return None, None
    
    def __contains__(self, key: str):
        return False
