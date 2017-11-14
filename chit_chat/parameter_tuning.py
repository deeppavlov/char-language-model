from environment import Environment
from residuals_no_authors_no_sampling import Lstm, LstmBatchGenerator
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

evaluation = dict(
    save_path='residuals_no_authors_no_sampling/parameter_tuning/1_tanh',
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets={'train': None,
              'default_1': [valid_text, 'default_1']},
    batch_gen_class=LstmBatchGenerator,
    batch_kwargs={'vocabulary': vocabulary},
    batch_size=1,
    additional_feed_dict=[{'placeholder': 'dropout', 'value': 1.}]
)

kwargs_for_building = dict(
          batch_size=64,
          num_layers=2,
          num_nodes=[1000, 1000],
          num_output_layers=2,
          num_output_nodes=[1000],
          vocabulary_size=vocabulary_size,
          connection_interval=3,
          subsequence_length_in_intervals=10,
          connection_visibility=9,
          init_parameter=3.,
          regularization_rate=.00001)


# env.grid_search(evaluation,
#                      kwargs_for_building,
#                      #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
#                      build_hyperparameters={'regularization_rate': [.000001, .000003, .00001, .00003, .0001, .0003, .001]},
#                      learning_rate={'type': 'exponential_decay', 'period': 1000, 'init': .002, 'decay': .9},
#                      batch_size=64,
#                      result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      vocabulary=vocabulary,
#                      stop=1000,
#                      num_unrollings=30,
#                      train_dataset_text=train_text,
#                      printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      results_collect_interval=100,
#                      validation_dataset_texts=[valid_text],
#                      no_validation=False,
#                      additions_to_feed_dict=[{'placeholder': 'dropout', 'value': {'type': 'fixed', 'name': 'dropout', 'value': 1.}}])

env.grid_search(evaluation,
                     kwargs_for_building,
                     build_hyperparameters={'init_parameter': [1., 1.3, 1.7, 2., 2.5, 3., 3.5, 4.]},
                     #other_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
                other_hyperparameters={'learning_rate': {'varying': {'init': [.0005, .0007, .001, .0013, .0015, .0017, .002, .0024, .003]},
                                                         'fixed': {'decay': 1., 'period': 1000},
                                                         'hp_type': 'built-in',
                                                         'type': 'exponential_decay'}},
                     #learning_rate={'type': 'exponential_decay', 'period': 1000, 'init': .002, 'decay': .9},
                     batch_size=64,
                     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
                     vocabulary=vocabulary,
                     stop=1000,
                     num_unrollings=30,
                     train_dataset_text=train_text,
                     printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
                     results_collect_interval=999,
                     validation_dataset_texts=[valid_text],
                     additions_to_feed_dict=[{'placeholder': 'dropout', 'value': {'type': 'fixed', 'name': 'dropout', 'value': .9}}],
                     no_validation=True)