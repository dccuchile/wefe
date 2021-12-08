"""Base clase for Word Embedding contaniers based on multiple sources."""
from typing import Callable, Dict, Sequence, Union
import numpy as np


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

    def getWordEmbeddings(self, target: str, attribute: str, template: str):
        return None, None
    
    def __contains__(self, key: str):
        return False

class EmbeddingMultiSet():
    def __init__(self):
        self.rows = []
    
    def add(self,
            target:str,
            attribute:str,
            tclass:int,
            aclass:int,
            tvector:np.array,
            avector:np.array):
        if any((item['target'] == target and 
                item['attribute'] == attribute and
                item['tclass'] == tclass and
                item['aclass'] == aclass) for item in self.rows):
            return False
    
        tmp = {'target'    : target,
               'attribute' : attribute,
               'tclass'    : tclass,
               'aclass'    : aclass,
               'tvector'   : tvector,
               'avector'   : avector}
        self.rows.append(tmp)
        return True
    
    def getTargets(self, tclass, aclass):
        res = []
        for row in self.rows:
            if row['tclass'] == tclass and row['aclass'] == aclass:
                res.append( row['tvector'] )
        return res
    
    def getTargetsMean(self, tclass, aclass):
        vals = {};
        
        for row in self.rows:
            if row['tclass'] == tclass and row['aclass'] == aclass:
                key = row['target']
                if key not in vals:
                    vals[key]  = [row['tvector']]
                else:
                    vals[key].append(row['tvector'])
        for key in vals:
            vals[key] = np.mean(vals[key], dtype=np.float64, axis=0)
        return list(vals.values())
    
    def getAllTargets(self, target_template):
        vals    = []
        classes = list(range(1, target_template + 1))
        for c in classes:
            vals.append({})
        
        for row in self.rows:
            index = row['tclass'] - 1
            key   = row['target']
            
            if key not in vals[index]:
                vals[index][key]  = [row['tvector']]
            else:
                vals[index][key].append(row['tvector'])
        
        word_set = []
        embe_set = []
        for i in range(0, target_template):
            words = []
            embed = []
            for key in vals[i]:
                words.append(key)
                embed.append(np.mean(vals[i][key], dtype=np.float64, axis=0))
            word_set.append(words)
            embe_set.append(embed)
        return embe_set, word_set
        
        vals = {}
        for row in self.rows:
            if row['tclass'] == tclass and row['aclass'] == aclass:
                key = row['target']
                if key not in vals:
                    vals[key]  = [row['tvector']]
                else:
                    vals[key].append(row['tvector'])
        for key in vals:
            vals[key] = np.mean(vals[key], dtype=np.float64, axis=0)
        return list(vals.values())
    
    def getAllAttributes(self, aclass):
        res = []
        for row in self.rows:
            if row['aclass'] == aclass:
                res.append( row['avector'] )
        return res
    
    def getAttributes(self, tclass, aclass):
        res = []
        for row in self.rows:
            if row['tclass'] == tclass and row['aclass'] == aclass:
                res.append( row['avector'] )
        return res
    
    def getAttributeWords(self, aclass):
        res = []
        for row in self.rows:
            if row['aclass'] == aclass:
                res.append( row['attribute'] )
        return res
    
    def getAttributesMean(self, tclass, aclass):
        vals = {};
        
        for row in self.rows:
            if row['tclass'] == tclass and row['aclass'] == aclass:
                key = row['attribute']
                if key not in vals:
                    vals[key]  = [row['avector']]
                else:
                    vals[key].append(row['avector'])
        for key in vals:
            vals[key] = np.mean(vals[key], dtype=np.float64, axis=0)
        return list(vals.values())
    
    def getEmbeddings(self, target, attribute):
        res = next((item for item in cont.rows if (item['target'] == target and item['attribute'] == attribute)), None)
        if res == None:
            return None, None
        return res['tvector'], res['avector']
