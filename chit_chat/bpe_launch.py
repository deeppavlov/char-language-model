from environment import Environment
from some_useful_functions import get_positions_in_vocabulary
from lstm_par import Lstm

"""for launches with one hot punctuation"""
from bpe import BpeBatchGeneratorOneHot as BatchGenerator
from bpe import create_vocabularies_one_hot as create_vocabularies
MAX_NUM_PUNCTUATION_MARKS = 6

"""for launches with free punctuation"""
# from bpe import BpeBatchGenerator as BatchGenerator
# from bpe import create_vocabulary
# from some_useful_functions import get_positions_in_vocabulary

with open('datasets/scipop_v3.0/bpe_train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# different
offset = 190
valid_size = 2000
valid_text = text[offset:offset + valid_size]
# train_text = text[offset + valid_size:]
train_text = text[offset:]
train_size = len(train_text)

"""for launches with one hot punctuation"""
punc_marks = list('!"\'(),-.:;? ')
tmp = create_vocabularies(text, punc_marks)
vocabulary_sizes = [len(voc) for voc in tmp]
print('vocabulary_sizes:', vocabulary_sizes)
word_voc, punc_voc = tmp
# print('word_voc:', word_voc)
print('punc_voc:', punc_voc)
word_cpiv = get_positions_in_vocabulary(word_voc)
punc_cpiv = get_positions_in_vocabulary(punc_voc)
# print('word_cpiv:', word_cpiv)
# print('punc_cpiv:', punc_cpiv)
env = Environment(Lstm, BatchGenerator, vocabulary=(word_voc, punc_voc))

"""for launches with free punctuation"""
# vocabulary = create_vocabulary(text)
# vocabulary_size = len(vocabulary)
# cpiv = get_positions_in_vocabulary(vocabulary)
# env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

add_feed = [{'placeholder': 'dropout', 'value': 0.8}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.build(batch_size=256,
          num_layers=2,
          num_nodes=[150, 150],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_sizes[0],
          # vocabulary_size=vocabulary_size,
          embedding_size=512,
          num_unrollings=10,
          going_to_limit_memory=True,
          number_of_punctuation_marks=len(punc_marks),
          max_mark_num=MAX_NUM_PUNCTUATION_MARKS,
          num_gpus=1)

# env.add_hooks(tensor_names=tensor_names)
env.train(save_path='lstm_bpe/first',
          learning_rate={'type': 'exponential_decay',
                         'init': .002,
                         'decay': .2,
                         'period': 20000},
          additions_to_feed_dict=add_feed,
          validation_additions_to_feed_dict=valid_add_feed,
          batch_size=256,
          num_unrollings=10,
          vocabulary=[word_voc, punc_voc],
          # vocabulary=vocabulary,
          checkpoint_steps=1000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          # validation_tensor_schedule=valid_tensors,
          # train_tensor_schedule=train_tensors,
          stop=100000,
          # train_dataset_text='abx',
          # validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          # validation_dataset=[valid_text],
          results_collect_interval=1000,
          no_validation=False)

# env.build(batch_size=1,
#           num_layers=2,
#           num_nodes=[1500, 1500],
#           num_output_layers=2,
#           num_output_nodes=[1024],
#           vocabulary_size=vocabulary_sizes[0],
#           # vocabulary_size=vocabulary_size,
#           embedding_size=512,
#           num_unrollings=1,
#           going_to_limit_memory=True,
#           number_of_punctuation_marks=len(punc_marks),
#           max_mark_num=MAX_NUM_PUNCTUATION_MARKS,
#           regime='inference',
#           num_gpus=1)
#
# env.inference(restore_path='lstm_bpe/first/checkpoints/final',
#               log_path='lstm_bpe/first/debug_split_inference',
#               batch_generator_class=BatchGenerator,
#               character_positions_in_vocabulary=[word_cpiv, punc_cpiv],
#               # character_positions_in_vocabulary=cpiv,
#               vocabulary=[word_voc, punc_voc],
#               # vocabulary=vocabulary,
#               additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}],
#               bpe_codes='datasets/scipop_v3.0/codes.txt',
#               batch_gen_args={'punctuation_voc_size': vocabulary_sizes[1],
#                               'punctuation_marks': punc_voc},
#               gpu_memory=.1)