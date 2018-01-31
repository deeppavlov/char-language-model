import os

from environment import Environment
from some_useful_functions import get_positions_in_vocabulary
from lstm_par import Lstm

PUNC_MARKS = list('!"\'(),-.:;? ')

"""for launches with one hot punctuation"""
# from bpe import BpeBatchGeneratorOneHot as BatchGenerator
from bpe import BpeFastBatchGeneratorOneHot as BatchGenerator
from bpe import create_vocabularies_one_hot as create_vocabularies
MAX_NUM_PUNCTUATION_MARKS = 6

# with open('datasets/scipop_v3.0/bpe_train.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

print('reached file opening')
with open('datasets/all_scipop_bpe.txt', 'r') as f:
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

"""for launches with one hot punctuation"""
punc_marks = list('!"\'(),-.:;? ')
dataset_f_names = ['datasets/all_scipop_word_voc.txt', 'datasets/all_scipop_punc_voc.txt']

print('reached vocabulary creation')

if os.path.exists(dataset_f_names[0]) and os.path.exists(dataset_f_names[1]):
    with open(dataset_f_names[0], 'r') as f:
        t = f.read()
        word_voc = t.split('\t')
    with open(dataset_f_names[1], 'r') as f:
        t = f.read()
        punc_voc = t.split('\t')
    vocabulary_sizes = [len(word_voc), len(punc_voc)]
else:
    tmp = create_vocabularies(text, punc_marks)
    vocabulary_sizes = [len(voc) for voc in tmp]
    print('vocabulary_sizes:', vocabulary_sizes)
    word_voc, punc_voc = tmp

    with open('datasets/all_scipop_word_voc.txt', 'w') as f:
        for w_idx, w in enumerate(word_voc):
            f.write(w)
            if w_idx < len(word_voc) - 1:
                f.write('\t')

    with open('datasets/all_scipop_punc_voc.txt', 'w') as f:
        for p_idx, p in enumerate(punc_voc):
            f.write(p)
            if p_idx < len(punc_voc) - 1:
                f.write('\t')

# print('word_voc:', word_voc)
print('punc_voc:', punc_voc)
word_cpiv = get_positions_in_vocabulary(word_voc)
punc_cpiv = get_positions_in_vocabulary(punc_voc)
# print('word_cpiv:', word_cpiv)
# print('punc_cpiv:', punc_cpiv)
env = Environment(Lstm, BatchGenerator, vocabulary=(word_voc, punc_voc))


add_feed = [{'placeholder': 'dropout', 'value': 0.8}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.build(batch_size=256,
          embeddings_in_batch=False,
          num_layers=2,
          num_nodes=[1300, 1300],
          num_output_layers=2,
          num_output_nodes=[2048],
          vocabulary_size=vocabulary_sizes[0],
          embedding_size=512,
          num_unrollings=100,
          going_to_limit_memory=True,
          number_of_punctuation_marks=len(punc_marks),
          max_mark_num=MAX_NUM_PUNCTUATION_MARKS,
          num_gpus=1)

# env.add_hooks(tensor_names=tensor_names)
env.train(save_path='lstm_bpe_punc/compare_ngrams_bpe_char_31.01.18',
          # restore_path='lstm_bpe_punc/compare_ngrams_bpe_char_31.01.18/checkpoints/40000',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .2,
                         'period': 100000},
          additions_to_feed_dict=add_feed,
          validation_additions_to_feed_dict=valid_add_feed,
          batch_size=256,
          num_unrollings=100,
          vocabulary=[word_voc, punc_voc],
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
          results_collect_interval=5000,
          example_length=100,
          no_validation=False)

# print('reached build')
# env.build(batch_size=1,
#           embeddings_in_batch=False,
#           num_layers=2,
#           num_nodes=[2000, 2000],
#           num_output_layers=2,
#           num_output_nodes=[2048],
#           vocabulary_size=vocabulary_sizes[0],
#           embedding_size=512,
#           num_unrollings=1,
#           going_to_limit_memory=True,
#           number_of_punctuation_marks=len(punc_marks),
#           max_mark_num=MAX_NUM_PUNCTUATION_MARKS,
#           regime='inference',
#           num_gpus=1)
#
# env.inference(restore_path='lstm_bpe_punc/compare_ngrams_bpe_char_31.01.18/checkpoints/50000',
#               log_path='lstm_bpe_punc/compare_ngrams_bpe_char_31.01.18/dialogs_1',
#               batch_generator_class=BatchGenerator,
#               character_positions_in_vocabulary=[word_cpiv, punc_cpiv],
#               vocabulary=[word_voc, punc_voc],
#               additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}],
#               # bpe_codes='datasets/scipop_v3.0/codes.txt',
#               bpe_codes='datasets/all_scipop_codes.txt',
#               batch_gen_args={'punctuation_voc_size': vocabulary_sizes[1],
#                               'punctuation_marks': punc_voc},
#               gpu_memory=.1)