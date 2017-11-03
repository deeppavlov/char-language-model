
from environment import Environment
from vanilla import Vanilla, BatchGenerator
from lstm_go import Lstm, LstmBatchGenerator
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
#env = Environment(Vanilla, BatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)
# env.build(batch_size=64,
#           num_nodes=250,
#           vocabulary_size=vocabulary_size,
#           num_unrollings=10)

add_feed = [{'placeholder': 'dropout', 'value': 0.9}]
valid_add_feed = [{'placeholder': 'dropout', 'value': .001}]

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[100, 100],
          num_output_layers=2,
          num_output_nodes=[100],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=10)

env.train(save_path='debugging_environment/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .003,
                         'decay': .9,
                         'period': 500},
          additions_to_feed_dict=add_feed,
          validation_additions_to_feed_dict=valid_add_feed,
          batch_size=64,
          num_unrollings=10,
          vocabulary=vocabulary,
          checkpoint_steps=[100],
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          stop=1000,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          #validation_dataset=[valid_text],
          results_collect_interval=100)

# evaluation = dict(
#     save_path='simple_lstm/tuning',
#     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#     datasets={'train': None,
#               'default_1': [valid_text, 'default_1']},
#     batch_gen_class=BatchGenerator,
#     batch_kwargs={'vocabulary': vocabulary},
#     batch_size=1,
#     additional_feed_dict=None
# )
#
# kwargs_for_building = dict(
#           batch_size=64,
#           num_layers=2,
#           num_nodes=[300, 300],
#           vocabulary_size=vocabulary_size,
#           num_unrollings=30)
#
# #list_of_lr = [dict(type='exponential_decay', init=v, decay=.9, period=500) for v in [10., 5., 3., 1., .3]]
# list_of_lr = [dict(type='exponential_decay', init=v, decay=.9, period=500) for v in [5., 2., 1.]]
#
# env.several_launches(evaluation,
#                      kwargs_for_building,
#                      #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
#                      build_hyperparameters={'init_parameter': [.5, 1.]},
#                      other_hyperparameters={'learning_rate': list_of_lr},
#                      batch_size=64,
#                      result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      vocabulary=vocabulary,
#                      stop=200,
#                      num_unrollings=30,
#                      train_dataset_text=train_text,
#                      printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      results_collect_interval=100,
#                      validation_dataset_texts=[valid_text],
#                      no_validation=False,
#                      additional_feed_dict=None)