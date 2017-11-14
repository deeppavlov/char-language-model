from environment import Environment
from attention_no_authors_no_sampling_par import Lstm, LstmBatchGenerator
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

connection_interval = 8
connection_visibility = 5
subsequence_length_in_intervals = 10

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[1200, 1200],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          connection_interval=connection_interval,
          subsequence_length_in_intervals=subsequence_length_in_intervals,
          connection_visibility=connection_visibility,
          init_parameter=3.,
          num_gpus=2)

env.train(save_path='attention_no_authors_no_sampling/debug_parallel',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .5,
                         'period': 26000},
          batch_size=64,
          num_unrollings=subsequence_length_in_intervals*connection_interval,
          vocabulary=vocabulary,
          checkpoint_steps=20000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          stop=100000,
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          results_collect_interval=1000,
          additions_to_feed_dict=[{'placeholder': 'dropout', 'value': .9}],
          validation_additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}])
