import os

from environment import Environment
from some_useful_functions import get_positions_in_vocabulary
from lstm_par import Lstm

NUMBER_OF_CHARS_IN_NGRAMS = 2

from ngrams import create_vocabulary, NgramsBatchGenerator as BatchGenerator

# with open('datasets/scipop_v3.0/bpe_train.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

print('reached file opening')
with open('datasets/all_scipop.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('file is opened')
# different
offset = 190
valid_size = 20000
valid_text = text[offset:offset + valid_size]
# train_text = text[offset + valid_size:]
train_text = text[offset + valid_size:]
print('looking for spaces')
counter = 0
while train_text[counter] != ' ':
    counter += 1
train_text = train_text[counter:]
# print('valid_text:\n', valid_text)
counter = 1
while valid_text[-counter] != ' ':
    counter += 1
valid_text = valid_text[:-counter]
print('spaces are found')

train_size = len(train_text)


"""for launches with free punctuation"""
print('reached vocabulary creation')

voc_name = 'datasets/all_scipop_ngrams_voc.txt'
if os.path.exists(voc_name):
    with open(voc_name, 'r') as f:
        t = f.read()
        vocabulary = t.split('\t')
    vocabulary_size = len(vocabulary)
else:
    vocabulary = create_vocabulary(text)
    vocabulary_size = len(vocabulary)
    with open(voc_name, 'w') as f:
        for w_idx, w in enumerate(vocabulary):
            f.write(w)
            if w_idx < len(vocabulary) - 1:
                f.write('\t')


cpiv = get_positions_in_vocabulary(vocabulary)
env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)



add_feed = [{'placeholder': 'dropout', 'value': 0.8}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.build(batch_size=256,
          num_layers=2,
          num_nodes=[2000, 2000],
          num_output_layers=2,
          num_output_nodes=[2048],
          vocabulary_size=vocabulary_size,
          embedding_size=512,
          num_unrollings=100,
          going_to_limit_memory=True,
          num_gpus=2)

# env.add_hooks(tensor_names=tensor_names)
env.train(save_path='lstm_ngrams/huge_adam',
          # restore_path='lstm_ngrams/huge_adam/checkpoints/40000',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .2,
                         'period': 100000},
          additions_to_feed_dict=add_feed,
          validation_additions_to_feed_dict=valid_add_feed,
          batch_size=256,
          num_unrollings=100,
          vocabulary=vocabulary,
          checkpoint_steps=50000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          # validation_tensor_schedule=valid_tensors,
          # train_tensor_schedule=train_tensors,
          stop=300000,
          # train_dataset_text='abx',
          # validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          # validation_dataset=[valid_text],
          results_collect_interval=1000,
          example_length=100,
          no_validation=False)

# print('reached build')
# env.build(batch_size=1,
#           num_layers=2,
#           num_nodes=[2000, 2000],
#           num_output_layers=2,
#           num_output_nodes=[2048],
#           vocabulary_size=vocabulary_size,
#           embedding_size=512,
#           num_unrollings=1,
#           going_to_limit_memory=True,
#           regime='inference',
#           num_gpus=1)
#
# env.inference(restore_path='lstm_bpe/huge_free_sgd/checkpoints/260000',
#               log_path='lstm_bpe/huge_free_sgd/dialogs_1',
#               batch_generator_class=BatchGenerator,
#               character_positions_in_vocabulary=cpiv,
#               vocabulary=vocabulary,
#               additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}],
#               gpu_memory=.1)