"""A Word Embedding contanier based on gensim BaseKeyedVectors."""
import numpy as np
from typing import Callable, Dict, Sequence, Union
from wefe.models.base_model import BaseModel
from simpletransformers.language_representation import RepresentationModel


class WERepresentationModel(BaseModel):
    #
    #
    #
    def __init__(
        self, wv: RepresentationModel, name: str = None, vocab_prefix: str = None
    ):
        # Type checking
        if not isinstance(wv, RepresentationModel):
            raise TypeError(
                "wv should be an instance of gensim's RepresentationModel"
                ", got {}.".format(wv)
            )

        if not isinstance(name, (str, type(None))):
            raise TypeError("name should be a string or None, got {}.".format(name))

        if not isinstance(vocab_prefix, (str, type(None))):
            raise TypeError(
                "vocab_prefix should be a string or None, got {}".format(vocab_prefix)
            )

        # Assign the attributes
        BaseModel.__init__(self, name, vocab_prefix)
        self.wv      = wv
        self.vocab   = None
        self.context = True
    
    #
    #
    #
    def __getitem__(self, key: str) -> Union[np.ndarray, None]:
        embedding = self.wv.encode_sentences([key], combine_strategy='mean')
        return embedding[0]
    
    #
    #
    #
    def getWordEmbeddings(self, target: str, attribute: str, template: str):
        template = template.replace('[TARGET]',    target)
        template = template.replace('[ATTRIBUTE]', attribute)
        
        tokens = self.wv.tokenizer([target])
        target_len = len(tokens['input_ids'][0]) - 2
        
        tokens = self.wv.tokenizer([attribute])
        attribute_len = len(tokens['input_ids'][0]) - 2
        
        embeddings = self.wv.encode_sentences([template], combine_strategy=None)
        embeddings_len = len(embeddings[0])
        
        we1 = embeddings[0][1:target_len+1]
        we2 = embeddings[0][embeddings_len - 1 - attribute_len:embeddings_len-1]
        return np.mean(we1, dtype=np.float64, axis=0), np.mean(we2, dtype=np.float64, axis=0)
