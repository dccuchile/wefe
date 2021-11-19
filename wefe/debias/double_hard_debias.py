import operator
from typing import Dict, Any, Optional, List
from numpy.core.fromnumeric import transpose
from wefe import debias
from wefe.debias.base_debias import BaseDebias
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from wefe.preprocessing import get_embeddings_from_sets
import numpy as np
from sklearn.metrics import precision_score
from scipy.spatial import distance
from wefe.word_embedding_model import WordEmbeddingModel

class DoubleHardDebias(BaseDebias):
    """Double Hard Debias Method.
    
    References
    ----------
    | [1]: Wang, Tianlu, Xi Victoria Lin, Nazneen Fatema Rajani, Bryan McCann, Vicente Or-donez y Caiming Xiong:
    | Double-Hard Debias: Tailoring Word Embeddings for GenderBias Mitigation. CoRR, abs/2005.00965, 
    | 2020.https://arxiv.org/abs/2005.00965.
    | [2]: https://github.com/uvavision/Double-Hard-Debias
    """
    
    name = "Double Hard Debias"
    short_name = "DHD"
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def similarity(self, u:np.ndarray, v:np.ndarray)->float:
        return 1-distance.cosine(u,v)
    
    def bias_by_projection(self, model:WordEmbeddingModel, exclude:List[str])->Dict[str,float]:
        he = model['he']
        she = model['she']
        similarities = {}
        for word in model.vocab:
            if word in exclude:
                continue
            embedding = model[word]
            similarities[word] = self.similarity(embedding,he) - self.similarity(embedding,she)
        return similarities

    def get_target_words(
        self, 
        model:WordEmbeddingModel,
        exclude:List[str],
        n_words:int
        ):
        similarities = self.bias_by_projection(model,exclude)
        sorted_words = sorted(similarities.items(), key=operator.itemgetter(1))
        female_words = [pair[0] for pair in sorted_words[:n_words]]
        male_words = [pair[0] for pair in sorted_words[:-n_words]]
        return female_words, male_words

    def principal_components(self, model:WordEmbeddingModel) -> np.ndarray:
        pca = IncrementalPCA() #PCA(svd_solver='randomized')
        pca.fit(model.wv.vectors - self.embeddings_mean)
        return pca.components_
    
    def calculate_embeddings_mean(self, model:WordEmbeddingModel) -> float:
        return np.mean(model.wv.vectors)
        
    def drop_frecuency_features(
        self, 
        components:int,
        model:WordEmbeddingModel
        )->Dict[str, np.ndarray]:
        
        droped_frecuencies = {}
        
        for word in self.target_words:
            embedding = model[word]
            decendecentralize_embedding = embedding- self.embeddings_mean
            frecuency = np.zeros(embedding.shape).astype(float)
            
            #for u in self.pca[components]:
            u = self.pca[components]
            frecuency = np.dot(np.dot(np.transpose(u),embedding),u)
            new_embedding = decendecentralize_embedding - frecuency
            
            droped_frecuencies[word] = new_embedding
        return droped_frecuencies
    
    def _identify_bias_subspace(
        self, 
        definning_pairs_embeddings, 
        verbose: bool = False, 
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

        pca = PCA() #PCA(**self.pca_args)
        pca.fit(matrix)

        if verbose:
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA variance explained: {explained_variance[0:pca.n_components_]}")

        return pca
    
    def drop(self,u:np.ndarray,v:np.ndarray)->np.ndarray:
        return u - v * u.dot(v) / v.dot(v)

    def debias(
        self, 
        words_dict:Dict[str, np.ndarray]
        )->Dict[str, np.ndarray]:
        
        for word in words_dict:
            embedding = words_dict[word]
            debias_embedding = self.drop(embedding,self.bias_direction)
            words_dict.update({word:debias_embedding})
        return words_dict
    
    def get_optimal_dimension(self, model:WordEmbeddingModel, n_words:int,n_components:int)->int:
        n_components = n_components
        scores =  []
        for d in range(n_components):
            result_embeddings = self.drop_frecuency_features(d,model)
            result_embeddings = self.debias(result_embeddings)
            y_true = [0]*n_words + [1]*n_words 
            scores.append(self.kmeans_eval(result_embeddings,y_true,n_words)) 
        min_precision = min(scores)
        
        return scores.index(min_precision)

    def kmeans_eval(
        self, 
        embeddings_dict:Dict[str, np.ndarray],
        y_true:List[int], 
        n_words: int,
        n_cluster:int=2
        )->float:     

        embeddings = [embeddings_dict[word] for word in self.target_words[0:2*n_words]]
        kmeans = KMeans(n_cluster).fit(embeddings)
        y_pred = kmeans.predict(embeddings)
        precision = precision_score(y_true,y_pred)
        return precision
    
    def fit(
        self,
        model:WordEmbeddingModel,
        defnitional_pairs
        )->BaseDebias:

        self.definitional_pairs = defnitional_pairs
        self.definitional_pairs_embeddings = get_embeddings_from_sets(model=model, sets=defnitional_pairs)
        self.bias_direction = self._identify_bias_subspace(self.definitional_pairs_embeddings).components_[0]
        self.embeddings_mean = self.calculate_embeddings_mean(model)
        self.pca = self.principal_components(model)
        
        return self
    
    def transform(
        self,
        model:WordEmbeddingModel, 
        exclude_words: List[str] = [],
        n_words: int = 1000,
        copy: bool = True,
        n_components: int = 4
        )->WordEmbeddingModel:
        
        female, male = self.get_target_words(model,exclude_words,n_words)

        target = female + male + sum(self.definitional_pairs,[]) 

        self.target_words = target

        optimal_dimensions = self.get_optimal_dimension(model,n_words,n_components)
  
        debiased_embeddings = self.drop_frecuency_features(optimal_dimensions,model)
        debiased_embeddings = self.debias(debiased_embeddings)
  
        for word in debiased_embeddings:
            model.update(word,debiased_embeddings[word].astype(model.wv.vectors.dtype))
            
        return model