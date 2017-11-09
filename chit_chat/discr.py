
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
test_lines = lines[int(.95*num_lines):]
valid_text = ''.join(used_lines[:100])
train_text = ''.join(used_lines[100:])
test_text = ''.join(test_lines)


# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[1700, 1700],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=100,
          num_gpus=2)

env.generate_discriminator_dataset(1000000, 1, test_text, 200, 'new_line',
                                   'lstm/big_network_1700_ted_correct_8.11/checkpoints/final',
                                   #'debugging_environment/first/checkpoints/final',
                                   'lstm/big_network_1700_ted_correct_8.11/final_discr',
                                   #'debugging_environment/first/final_discr',
                                   additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}])
