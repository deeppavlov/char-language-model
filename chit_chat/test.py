
from environment import Environment
from lstm_sample_par import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/scipop_train.txt', 'r', encoding='utf-8')
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

env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)


add_feed = [{'placeholder': 'dropout', 'value': 0.9},
            {'placeholder': 'sampling_prob', 'value': {'type': 'linear', 'start': 0., 'end': 1., 'interval': 3000}},
            {'placeholder': 'loss_comp_prob', 'value': {'type': 'linear', 'start': 1., 'end': 0., 'interval': 3000}}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.},
                  {'placeholder': 'sampling_prob', 'value': 1.},
                  {'placeholder': 'loss_comp_prob', 'value': .0}]

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[170, 170],
          num_output_layers=2,
          num_output_nodes=[124],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=10,
          character_positions_in_vocabulary=cpiv)

env.train(save_path='debugging_environment/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
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
          stop=5000,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          #validation_dataset=[valid_text],
          results_collect_interval=100)

connection_interval = 8
connection_visibility = 5
subsequence_length_in_intervals = 10

# env.build(batch_size=64,
#           num_layers=2,
#           num_nodes=[200, 200],
#           num_output_layers=2,
#           num_output_nodes=[200],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           #dim_compressed=4,
#           connection_interval=connection_interval,
#           subsequence_length_in_intervals=subsequence_length_in_intervals,
#           connection_visibility=connection_visibility)

# tensor_names = [('sample_state_0', 'validation/saved_sample_from_connections_0:0'),
#                 ('for_conn', 'validation/saved_sample_state:0')]
# for idx in range(connection_visibility):
#     tensor_names.append(('saved_for_connection_%s' % idx, 'train/saved_for_connection_%s:0' % idx))
# env.add_hooks(tensor_names=tensor_names)
#
# valid_tensors = {'valid_print_tensors': {'sample_state_0': [i for i in range(100)],
#                                          'for_conn': [i for i in range(100)]}}
# train_print = dict()
# for idx in range(connection_visibility):
#     train_print['saved_for_connection_%s' % idx] = [i for i in range(100)]
# train_tensors = {'train_print_tensors': train_print}

# env.train(save_path='debugging_environment/first',
#           learning_rate={'type': 'exponential_decay',
#                          'init': .002,
#                          'decay': .5,
#                          'period': 20000},
#           additions_to_feed_dict=add_feed,
#           validation_additions_to_feed_dict=valid_add_feed,
#           batch_size=64,
#           num_unrollings=subsequence_length_in_intervals*connection_interval,
#           vocabulary=vocabulary,
#           checkpoint_steps=1000,
#           result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           stop=10000,
#           #validation_tensor_schedule=valid_tensors,
#           #train_tensor_schedule=train_tensors,
#           #train_dataset_text='abx',
#           #validation_datasets_texts=['abc'],
#           train_dataset_text=train_text,
#           validation_dataset_texts=[valid_text],
#           #validation_dataset=[valid_text],
#           results_collect_interval=1000)

# evaluation = dict(
#     save_path='simple_lstm/tuning2',
#     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#     datasets={'train': None,
#               'default_1': [valid_text, 'default_1']},
#     batch_gen_class=BatchGenerator,
#     batch_kwargs={'vocabulary': vocabulary},
#     batch_size=1,
#     additions_to_feed_dict=[{'type': 'fixed', 'name': 'dropout', 'value': 1.}]
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
#
# env.grid_search(evaluation,
#                      kwargs_for_building,
#                      #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
#                      build_hyperparameters={'num_nodes[0,1]': [[250, 350, 450]]*2},
#                      # other_hyperparameters={'learning_rate': {'hp_type': 'built-in',
#                      #                                          'type': 'exponential_decay',
#                      #                                          'fixed': {'period': 500, 'decay': .9},
#                      #                                          'varying': {'init': [.002, .001, .0015]}}},
#                      batch_size=64,
#                      result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      vocabulary=vocabulary,
#                      stop=50,
#                      num_unrollings=30,
#                      train_dataset_text=train_text,
#                      printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      results_collect_interval=25,
#                      validation_dataset_texts=[valid_text],
#                      no_validation=False,
#                      additions_to_feed_dict=[{'type': 'fixed', 'name': 'dropout', 'value': 1.}])

# env.build(batch_size=64,
#           num_layers=2,
#           num_nodes=[200, 200],
#           num_output_layers=2,
#           num_output_nodes=[200],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           #dim_compressed=4,
#           connection_interval=connection_interval,
#           subsequence_length_in_intervals=subsequence_length_in_intervals,
#           connection_visibility=connection_visibility,
#           regime='inference')

# env.inference(restore_path='debugging_environment/first/checkpoints/1000',
#               log_path='debugging_environment/first/debug_split_inference',
#               batch_generator_class=LstmBatchGenerator,
#               character_positions_in_vocabulary=cpiv,
#               vocabulary=vocabulary,
#               additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1}],
#               gpu_memory=.1)

# env.generate_discriminator_dataset(10000000, 1, text, 200, 'new_line', 'debugging_environment/first/checkpoints/1000',
#                                    'debugging_environment/first/debug_gen_dataset',
#                                    additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}], gpu_memory=.1)