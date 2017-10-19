from environment import Environment
from lstm import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary


f = open('enwik8_clean', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
offset = 10000
valid_size = 10000
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
          num_nodes=[2048],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=50,
          init_parameter=4.)

env.train(save_path='debugging_environment/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .005,
                         'decay': .9,
                         'period': 500},
          batch_size=64,
          num_unrollings=50,
          vocabulary=vocabulary,
          checkpoint_steps=20000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          stop=100000,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          #validation_dataset=[valid_text],
          results_collect_interval=400)