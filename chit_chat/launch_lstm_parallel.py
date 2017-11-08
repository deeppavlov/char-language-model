from environment import Environment
from lstm_go_par import Lstm, LstmBatchGenerator
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
          num_nodes=[1900, 1900],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=100,
          init_parameter=3.,
          num_gpus=2)

env.train(save_path='lstm_go/big_network_ted_correct_8.11',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .5,
                         'period': 26000},
          batch_size=64,
          num_unrollings=100,
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
