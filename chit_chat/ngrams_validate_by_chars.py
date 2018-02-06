import os

from environment import Environment
from some_useful_functions import get_positions_in_vocabulary
from lstm_par import Lstm

NUMBER_OF_CHARS_IN_NGRAMS = 2

from ngrams import create_vocabulary
# from ngrams import NgramsBatchGenerator as BatchGenerator
from ngrams import NgramsFastBatchGenerator as BatchGenerator

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

print('reached build')
env.build(batch_size=1,
          embeddings_in_batch=False,
          num_layers=2,
          num_nodes=[1300, 1300],
          num_output_layers=2,
          num_output_nodes=[2048],
          vocabulary_size=vocabulary_size,
          embedding_size=512,
          num_unrollings=1,
          going_to_limit_memory=True,
          regime='inference',
          num_gpus=1)

env.test(
    restore_path='lstm_ngrams/compare_ngrams_bpe_char_31.01.18/checkpoints/final',
    save_path='lstm_ngrams/compare_ngrams_bpe_char_31.01.18/test/final',
    vocabulary=vocabulary,
    additions_to_feed_dict=valid_add_feed,
    validation_dataset_texts=[valid_text],
    validate_tokens_by_chars=True,
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
)