from environment import Environment
import re
# from residuals_no_authors_no_sampling import Lstm, LstmBatchGenerator
from lstm_par import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/scipop_v3.0/scipop_train.txt', 'r', encoding='utf-8')
text = f.read()
text = re.sub('<[^>]*>', '', text)
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

env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)
cpiv = get_positions_in_vocabulary(vocabulary)

kwargs_for_building = dict(
          batch_size=64,
          num_layers=2,
          num_nodes=[300, 300],
          num_output_layers=2,
          num_output_nodes=[124],
          vocabulary_size=vocabulary_size,
          # dim_compressed=10,
          num_unrollings=30,
          init_parameter=3.,
          regularization_rate=.00001,
          regime='inference',
          going_to_limit_memory=True)

add_feed = [{'placeholder': 'dropout', 'value': 0.9}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.telegram(
    kwargs_for_building,
    'debugging_lstm_and_gru/first/checkpoints/100',
    'telegram/debug',
    vocabulary,
    cpiv,
    LstmBatchGenerator,
    additions_to_feed_dict=valid_add_feed)
