"""
===========================
Basic usage of WEFE with WEAT and word2vec
===========================

An example of basic usage of wefe with weat and word2vec :class:`wefe.metrics.WEAT`
"""

from wefe.datasets.datasets import load_weat, fetch_bingliu
from wefe.metrics.WEAT import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

import gensim.downloader as api

# Load a pretrained word embedding.
w2v = api.load('word2vec-google-news-300')

# Create the model object with the embeddings.
model = WordEmbeddingModel(w2v, 'word2vec-google-news')

# Load a word set. In this case we will use the preloaded weat_word_set
weat_word_set = load_weat()

flowers = weat_word_set['Flowers']  # ['aster','clover','hyacinth','marigold',...]
insects = weat_word_set['Insects']  # ['ant','caterpillar','flea','locust',...]
pleasant = weat_word_set['Pleasant 5']  # ['caress','freedom','health','love',...]
unpleasant = weat_word_set['Unpleasant 5']  # ['abuse','crash','filth','murder',...]

# Create the query. In this case, we will use two target set and two attribute sets:
# First, we specify the arrays that will use as target and attribute word sets.
# Then, we specify a their correspondant names.
query = Query([flowers, insects], [pleasant, unpleasant], ['Flowers', 'Insects'], ['Pleasant', 'Unpleasant'])

# Instance a Metric. In this case, WEAT:
weat = WEAT()

# Run the experiment:
results = weat.run_query(query, model, warn_filtered_words=True, return_effect_size=True)
print(results)