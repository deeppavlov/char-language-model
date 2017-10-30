from environment import Environment
from lstm_par import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/ted.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
offset = 10000
valid_size = 1000
valid_text = text[offset:offset + valid_size]
train_text = text[offset + valid_size:]
train_size = len(train_text)

# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[100, 103],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=10)

env.train(save_path='debugging_parallel/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .003,
                         'decay': .9,
                         'period': 500},
          batch_size=64,
          num_unrollings=10,
          vocabulary=vocabulary,
          checkpoint_steps=[100],
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          stop=1000,
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          results_collect_interval=100)
