from environment import Environment
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
cpiv = get_positions_in_vocabulary(vocabulary)

evaluation = dict(
    save_path='lstm/tuning_reg_rate',
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets={'train': None,
              'default_1': [valid_text, 'default_1']},
    batch_gen_class=LstmBatchGenerator,
    batch_kwargs={'vocabulary': vocabulary},
    batch_size=1,
    additional_feed_dict=None
)

kwargs_for_building = dict(
          batch_size=64,
          num_layers=2,
          num_nodes=[2000, 2000],
          num_output_layers=2,
          num_output_nodes=[1000],
          vocabulary_size=vocabulary_size,
          num_unrollings=30,
          init_parameter=3.)


env.grid_search(evaluation,
                     kwargs_for_building,
                     #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
                     build_hyperparameters={'regularization_rate': [.000001, .000003, .00001, .00003, .0001, .0003, .001]},
                     learning_rate={'type': 'exponential_decay', 'period': 1000, 'init': .002},
                     batch_size=64,
                     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
                     vocabulary=vocabulary,
                     stop=1000,
                     num_unrollings=30,
                     train_dataset_text=train_text,
                     printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
                     results_collect_interval=100,
                     validation_dataset_texts=[valid_text],
                     no_validation=False,
                     additional_feed_dict=None)

# env.grid_search(evaluation,
#                      kwargs_for_building,
#                      #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
#                      build_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
#                      learning_rate={'type': 'exponential_decay', 'period': 1000, 'init': .002},
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
#                      additional_feed_dict=None)