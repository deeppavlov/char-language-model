from environment import Environment
from lstm import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary


f = open('datasets/ted.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
lines = text.splitlines(True)
num_lines = len(lines)
used_lines = lines[:int(.95*num_lines)]
valid_text = ''.join(used_lines[:100])
train_text = ''.join(used_lines[100:])

# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[1048, 1048],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=50,
          init_parameter=3.)

env.train(save_path='debugging_environment/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .5,
                         'period': 15000},
          batch_size=64,
          num_unrollings=100,
          vocabulary=vocabulary,
          checkpoint_steps=10000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          stop=100000,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          #validation_dataset=[valid_text],
          results_collect_interval=1000,
          visible_device_list="3")