from wefe.datasets.datasets import load_weat
from wefe.metrics.WEAT import WEAT
from wefe.utils import load_weat_w2v
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

weat_word_set = load_weat()
model = WordEmbeddingModel(load_weat_w2v(), 'weat_w2v', '')

weat = WEAT()
queries = [
    Query([weat_word_set['Flowers'], weat_word_set['Insects']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
          ['Pleasant', 'Unpleasant']),
    Query([weat_word_set['Instruments'], weat_word_set['Weapons']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Instruments', 'Weapons'],
          ['Pleasant', 'Unpleasant']),
    Query([weat_word_set['Flowers'], weat_word_set['Insects']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
          ['Pleasant', 'Unpleasant']),
    Query([weat_word_set['Flowers'], weat_word_set['Insects']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
          ['Pleasant', 'Unpleasant']),
    Query([weat_word_set['Flowers'], weat_word_set['Insects']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
          ['Pleasant', 'Unpleasant']),
    Query([weat_word_set['Flowers'], weat_word_set['Insects']],
          [weat_word_set['Pleasant 5'], weat_word_set['Unpleasant 5']], ['Flowers', 'Insects'],
          ['Pleasant', 'Unpleasant']),
]

results = weat.run_query(query, model, warn_filtered_words=True, return_effect_size=True)
