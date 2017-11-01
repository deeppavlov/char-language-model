
from environment import Environment
from vanilla import Vanilla, BatchGenerator
from lstm_go_par import Lstm, LstmBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/ted.txt', 'r', encoding='utf-8')
text = f.read()
f.close()


# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[1700, 1700],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=10,
          num_gpus=1)

env.inference(restore_path='lstm_go/nu_100_ns200k_nl2_nn1700/checkpoints/40000',
              log_path='lstm_go/nu_100_ns200k_nl2_nn1700/dialogs/1',
              vocabulary=vocabulary,
              characters_positions_in_vocabulary=cpiv,
              batch_generator_class=LstmBatchGenerator,
              temperature=1.)