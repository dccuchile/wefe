===========================
WEFE Case Study Replication
===========================

The following code replicates the case study presented in our paper: 

P. Badilla, F. Bravo-Marquez, and J. Pérez 
WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
29th International Joint Conference on Artificial Intelligence and the 17th 
Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. 


In this study we evaluate:

- Multiple queries grouped according to different criteria (gender, ethnicity, religion)
- Multiple embeddings (:code:`word2vec-google-news`, :code:`glove-wikipedia`, 
  :code:`glove-twitter`, :code:`conceptnet`, :code:`lexvec`, 
  :code:`fasttext-wiki-news`)
- Multiple metrics (:code:`WEAT` and its variant, :code:`WEAT effect size`, 
  :code:`RND`, :code:`RNSB`). 

After grouping the results by each criterion and metric, the rankings of the 
bias scores of each embedding model are calculated and plotted. 
An overall ranking is also computed, which is simply the sum of all rankings 
by model and metric.

Finally, the matrix of correlations between these rankings is calculated and 
plotted.

The code for this experiment is relatively long to run.
A Jupyter Notebook with the code is provided in the 
following `link <https://github.com/dccuchile/wefe/blob/master/examples/WEFE_rankings.ipynb>`_.
