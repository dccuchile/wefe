from codecs import ignore_errors
import operator
from copy import deepcopy
from tabnanny import verbose
from typing import Dict, Any, Optional, List, Sequence
from wefe import debias
from wefe.debias.base_debias import BaseDebias
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from wefe.preprocessing import EmbeddingDict, get_embeddings_from_sets
import numpy as np
from wefe.utils import check_is_fitted
from wefe.word_embedding_model import WordEmbeddingModel
import torch.nn as nn
import torch
from copy import deepcopy


class RepulsionattractionNeutralization(BaseDebias):
    """Repulsion attraction Neutralization method.


    References
    ----------
    | [1]: Kumar, Vaibhav, Tenzin Singhay Bhotia y Tanmoy Chakraborty: Nurse is Closer to Wo-
        man than Surgeon? Mitigating Gender-Biased Proximities in Word Embeddings. CoRR,
        abs/2006.01938, 2020. https://arxiv.org/abs/2006.01938
    | [2]: https://github.com/TimeTraveller-San/RAN-Debias
    """    
    name = "Repulsion attraction Neutralization"
    short_name = "RAN"
    
    
    def __init__(
        self,
        pca_args: Dict[str, Any] = {"n_components": 10},
        verbose: bool = False,
        criterion_name: Optional[str] = None,
        ) -> None:
        """Initialize a Half Sibling Regression Debias instance.

        Parameters
        ----------
        verbose : bool, optional
            True will print informative messages about the debiasing process,
            by default False.
        criterion_name : Optional[str], optional
            The name of the criterion for which the debias is being executed,
            e.g., 'Gender'. This will indicate the name of the model returning transform,
            by default None
        """
        # check verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose should be a bool, got {verbose}.")
        self.pca_args = pca_args
        self.verbose = verbose
        
        if criterion_name is None or isinstance(criterion_name, str):
            self.criterion_name_ = criterion_name
        else:
            raise ValueError(f"criterion_name should be str, got: {criterion_name}")

    def _identify_bias_subspace(
        self, definning_pairs_embeddings: List[EmbeddingDict], verbose: bool = False,
    ) -> PCA:

        matrix = []
        for embedding_dict_pair in definning_pairs_embeddings:

            # Get the center of the current definning pair.
            pair_embeddings = np.array(list(embedding_dict_pair.values()))
            center = np.mean(pair_embeddings, axis=0)
            # For each word, embedding in the definning pair:
            for embedding in embedding_dict_pair.values():
                # Substract the center of the pair to the embedding
                matrix.append(embedding - center)
        matrix = np.array(matrix)  # type: ignore

        pca = PCA(**self.pca_args)
        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA variance explained: {explained_variance[0:pca.n_components_]}")

        return pca
    
    def indirect_bias(self, w, v, bias_direction):
        """
        computes indirect bias
        """        
        wv = np.dot(w,v) ##NORMALIZAR LOS VECTORS???
        w_orth = w - np.dot(w,bias_direction)*bias_direction
        v_orth = v - np.dot(v,bias_direction)*bias_direction
        cos_wv_orth = np.dot(w_orth,v_orth) / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))
        bias = (wv - cos_wv_orth) / wv
        return bias 
    
    def get_neighbours(self, model, word, n_neighbours = 100):
        """
        returns the n words closer to a given word
        """        
        similar_words = model.wv.most_similar(positive=word,topn=n_neighbours)
        similar_words = list(list(zip(*similar_words))[0])
        return similar_words
    
    def get_repulsion_set(self, model, word, bias_direction, theta, n_neighbours):
        """

        Repulsion set for given word. Dictionary with words in set and their vectors
        """        
        neighbours = self.get_neighbours(model, word, n_neighbours)
        repulsion_set = []
        for neighbour in neighbours:
            if self.indirect_bias(model[neighbour], model[word], bias_direction) > theta:
                repulsion_set.append(model[neighbour])
        return repulsion_set
    
    def get_all_repulsion(self, model, target_words, n_neighbours, bias_direction, theta):
        """
        gets repulsions sets and non repulsion sets for all target words
        """        
        all_repulsion = {}
        for word in target_words:
            neighbours = self.get_neighbours(model, word, n_neighbours)
            repulsion = self.get_repulsion_set(model, word, neighbours, bias_direction, theta)
            all_repulsion[word] = repulsion 
        return all_repulsion
            
    def cosine_similarity(self, w_b, set_vectors):
        return torch.matmul(set_vectors, w_b) / (set_vectors.norm(dim=1) * w_b.norm(dim=0))
    
    def repulsion(self, w_b, repulsion_set):
        if not isinstance(repulsion_set, bool):
            cos_similarity = self.cosine_similarity(w_b, repulsion_set)
            repulsion = torch.abs(cos_similarity).mean(dim=0)
        else:
            repulsion = 0
        return repulsion
    
    def attraction(self,w_b,w):
        attraction = torch.abs(torch.cosine_similarity(w_b[None,:],w[None,:]) - 1)[0]/2 
        return attraction
    
    def neutralization(self,w_b,bias_direction):
        neutralization = torch.abs(w_b.dot(bias_direction)).mean(dim=0)
        return neutralization 
    
    
    def objective_function(self,w_b,w,bias_direction,repulsion_set,weights):
        w1,w2,w3 = weights
        return self.repulsion(w_b,repulsion_set)*w1 + self.attraction(w_b,w)*w2 + self.neutralization(w_b,bias_direction)*w3 #W3????**
    
  
   
    def debias(self, model, word, w, w_b, bias_direction, repulsion_set, learning_rate, epochs):
        ran = RAN( model, word,w_b, w, repulsion_set, bias_direction, self.objective_function, weights=[0.33, 0.33, 0.33])
        optimizer = torch.optim.Adam(ran.parameters(), lr=learning_rate) ##dejar el optimizador como parametro?
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = ran.forward()
            out.backward()
            optimizer.step()
        debiased_vector = ran.w_b
        return debiased_vector #normalizar????
    
    def init_vector(self,model, word):
        v = deepcopy(model[word]) 
        return torch.FloatTensor(np.array(v))
    
    def fit(
        self,
        model:WordEmbeddingModel,
        definitional_pairs: Sequence[Sequence[str]],
        alpha: float = 60,
        theta: float = 0.04
        )-> BaseDebias:
        """
        Computes the weight matrix and the gender information
        
        Parameters
        ----------
    
        model: WordEmbeddingModel
            The word embedding model to debias. 
            
        gender_definition: Sequence[str]
            List of strings. This list contains words that embody gender information by definition.
            
        alpha: float
            Ridge Regression constant. By default 60,
            numner

        Returns
        -------
        BaseDebias
            The debias method fitted.
        """            
        
        # Check arguments types
        #self._check_sets_size(definitional_pairs, "definitional")
        self.definitional_pairs_ = definitional_pairs

        # ------------------------------------------------------------------------------
        # Obtain the embedding of each definitional pairs.
        if self.verbose:
            print("Obtaining definitional pairs.")
        self.definitional_pairs_embeddings_ = get_embeddings_from_sets(
            model=model,
            sets=definitional_pairs,
            sets_name="definitional",
            warn_lost_sets=self.verbose,
            normalize=True,
            verbose=self.verbose,
        )

        # ------------------------------------------------------------------------------:
        # Identify the bias subspace using the definning pairs.
        if self.verbose:
            print("Identifying the bias subspace.")

        self.pca_ = self._identify_bias_subspace(
            self.definitional_pairs_embeddings_, self.verbose,
        )
        self.bias_direction_ = self.pca_.components_[0]

        #self.repulsion = self.get_all_repulsion(model, list(model.vocab.keys()), 100, self.bias_direction_, theta )
        
        return self
    
    
    def transform(
        self,
        model:WordEmbeddingModel, 
        target: Optional[List[str]],
        ignore: Optional[List[str]] = [],
        learning_rate: float = 0.01, 
        epochs: int = 300,
        copy: bool = True,
        theta: float = 0.05,
        n_neighbours: int = 100
        )-> WordEmbeddingModel:
        
        """
        Substracts the gender information from vectors.
        
        Args:
            model: WordEmbeddingModel
                The word embedding model to debias.
                
            ignore (List[str], optional): _description_. Defaults to [].
            
            copy : bool, optional
                If `True`, the debias will be performed on a copy of the model.
                If `False`, the debias will be applied on the same model delivered, causing
                its vectors to mutate.
                **WARNING:** Setting copy with `True` requires RAM at least 2x of the size
                of the model, otherwise the execution of the debias may raise to
                `MemoryError`, by default True.

        WordEmbeddingModel
            The debiased embedding model.
        """    
        # check if the following attributes exist in the object.
       # check_is_fitted(
       #     self,
       #     [
       #     "*"
       #     ],
       # )
        
        # ------------------------------------------------------------------------------
        # Copy
        if copy:
            print(
                "Copy argument is True. Transform will attempt to create a copy "
                "of the original model. This may fail due to lack of memory."
            )
            model = deepcopy(model)
            print("Model copy created successfully.")

        else:
            print(
                "copy argument is False. The execution of this method will mutate "
                "the original model."
            )
        
        if self.verbose:
            print(f"Executing Repulsion attraction Neutralization Debias on {model.name}")
            
        debiased = {}
        for word in target:
            if word in ignore or word not in model:
                continue
            w = model[word]
            w_b = self.init_vector(model, word) 
            repulsion = self.get_repulsion_set(model, word, self.bias_direction_, theta, n_neighbours)
            new_embedding = self.debias(model, word, w, w_b, self.bias_direction_, repulsion, learning_rate, epochs)
            debiased[word] = new_embedding.detach().numpy()
            #model.update(word,new_embedding.astype(model.wv.vectors.dtype))

        for word in debiased:
            model.update(word,debiased[word].astype(model.wv.vectors.dtype))

        # ------------------------------------------------------------------------------
        # # Generate the new KeyedVectors
        if self.criterion_name_ is None:
            new_model_name = f"{model.name}_debiased"
        else:
            new_model_name = f"{model.name}_{self.criterion_name_}_debiased"
        model.name = new_model_name

        if self.verbose:
            print("Done!")
            
        return model
    
class RAN(nn.Module):
    def __init__(self, model, word,w_b, w, repulsion_set, bias_direction, objective_function, weights=[0.33, 0.33, 0.33]):
        super(RAN, self).__init__()

        self.model = model
        self.word = word
        self.w =  torch.FloatTensor(np.array(w)).requires_grad_(True)    
        #print(repulsion_set)     
        if len(repulsion_set) == 0:
            self.repulsion_set = False 
        else:
            self.repulsion_set = torch.FloatTensor(np.array(repulsion_set)).requires_grad_(True)
                
        self.w_b= nn.Parameter(w_b)
            
        self.bias_direction = torch.FloatTensor(np.array(bias_direction))
            
        self.weights = weights
            
        self.objective_function = objective_function

    def forward(self): 
        return  self.objective_function(self.w_b, self.w, self.bias_direction, self.repulsion_set, self.weights) #tiene que ir el repulsion set aca
        
      