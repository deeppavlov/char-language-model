from environment import Environment
from lstm import Lstm, LstmBatchGenerator
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

#list_of_lr = [dict(type='exponential_decay', init=v, decay=.9, period=500) for v in [10., 5., 3., 1., .3]]
list_of_lr = [dict(type='exponential_decay', init=v, decay=.9, period=500) for v in [.001, .0015, .002]]

env.several_launches(evaluation,
                     kwargs_for_building,
                     #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
                     build_hyperparameters={'regulariztion_rate': [.001, .01, .03, .1, .3, 1., 3., 10.]},
                     other_hyperparameters={'learning_rate': list_of_lr},
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