
from environment import Environment
from vanilla import Vanilla, BatchGenerator
from lstm_go import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/ted.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
offset = 0
valid_size = 20000
valid_text = text[offset:offset + valid_size]
train_text = text[offset + valid_size:]
train_size = len(train_text)

# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=50,
          num_layers=2,
          num_nodes=[2048, 2048],
          num_output_layers=2,
          num_output_nodes=[512],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=50)

env.generate_discriminator_dataset(2000, 10, valid_text, 200, 'new_line',
                                   'lstm_go_with_output/bs50_nl2_nol2_nn2048_non512_nu50_ilr.002_ip3.0/checkpoints/final',
                                   'debug_discr_data/first')
