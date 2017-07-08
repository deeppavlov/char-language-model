# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import random
import string
import tensorflow as tf
import tensorflow.python.ops.rnn_cell 
from tensorflow.python.framework import registry
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import collections
import matplotlib.pyplot as plt
import codecs
import time
import os
import gc
from six.moves import cPickle as pickle
import sys

from plot_module import text_plot
from plot_module import structure_vocabulary_plots
from plot_module import text_boundaries_plot
from plot_module import ComparePlots

from model_module import maybe_download
from model_module import read_data
from model_module import check_not_one_byte
from model_module import id2char
from model_module import char2id
from model_module import BatchGenerator
from model_module import characters
from model_module import batches2string
from model_module import logprob
from model_module import sample_distribution
from model_module import MODEL
from HM_LSTM import HM_LSTM

version = sys.version_info[0]


# second cell

if not os.path.exists('enwik8_filtered'):
    if not os.path.exists('enwik8'):
        filename = maybe_download('enwik8.zip', 36445475)
    full_text = read_data(filename)
    new_text = u""
    new_text_list = list()
    for i in range(len(full_text)):
        if (i+1) % 10000000 == 0:
            print("%s characters are filtered" % i)
        if ord(full_text[i]) < 256:
            new_text_list.append(full_text[i])
    text = new_text.join(new_text_list)
    del new_text_list
    del new_text
    del full_text

    (not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters_indices) = check_not_one_byte(text)

    print("number of not one byte characters: ", not_one_byte_counter) 
    print("min order index: ", min_character_order_index)
    print("max order index: ", max_character_order_index)
    print("total number of characters: ", number_of_characters)
    
    f = open('enwik8_filtered', 'wb')
    f.write(text.encode('utf8'))
    f.close()
    
else:
    f = open('enwik8_filtered', 'rb')
    text = f.read().decode('utf8')
    f.close() 
    (not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters_indices) = check_not_one_byte(text)

    print("number of not one byte characters: ", not_one_byte_counter) 
    print("min order index: ", min_character_order_index)
    print("max order index: ", max_character_order_index)
    print("total number of characters: ", number_of_characters)    

#third cell

#different
offset_1 = 0
offset_2 = 4100
valid_size_1 = 4000
valid_size_2 = 4000
valid_text_1 = text[offset_1:offset_1+valid_size_1]
valid_text_2 = text[offset_2:offset_2+valid_size_2]
train_text = text[offset_2+valid_size_2:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size_1, valid_text_1[:64])
print(valid_size_2, valid_text_2[:64])
print(valid_text_1)
print('\n\n\n')
print(valid_text_2)

# fourth cell

#different
offset = 20000
valid_size = 25000
valid_text = text[offset:offset+valid_size]
train_text = text[offset+valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

#fifth cell

vocabulary_size = number_of_characters
vocabulary = list()
characters_positions_in_vocabulary = list()

character_position_in_vocabulary = 0
for i in range(256):
    if present_characters_indices[i]:
        if version >= 3:
            vocabulary.append(chr(i))
        else:
            vocabulary.append(unichr(i))
        characters_positions_in_vocabulary.append(character_position_in_vocabulary)
        character_position_in_vocabulary += 1
    else:
        characters_positions_in_vocabulary.append(-1)


string_vocabulary = u""
for i in range(vocabulary_size):
    string_vocabulary += vocabulary[i]
print("Vocabulary: ", string_vocabulary)
print("char2id(u'a') = %s,  char2id(u'z') = %s,  char2id(u' ') = %s" % (char2id(u'a', characters_positions_in_vocabulary),
                                                                        char2id(u'z', characters_positions_in_vocabulary),
                                                                        char2id(u' ', characters_positions_in_vocabulary)))
print("id2char(78) = %s,  id2char(156) = %s,  id2char(140) = %s" % (id2char(78,
                                                                            vocabulary),
                                                                    id2char(156,
                                                                            vocabulary),
                                                                    id2char(140,
                                                                            vocabulary)))

# sixth xell

batch_size_test=64
num_unrollings_test=10

train_batches_test = BatchGenerator(train_text,
                                    batch_size_test,
                                    vocabulary_size,
                                    characters_positions_in_vocabulary,
                                    num_unrollings_test)
valid_batches_test = BatchGenerator(valid_text_1,
                                    1,
                                    vocabulary_size,
                                    characters_positions_in_vocabulary,
                                    1)

print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))

# seventh cell


model = HM_LSTM(53,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 30,
                 3,
                 [127, 89, 61],
                 .001,               # init_slope
                 0.001,                  # slope_growth
                 100,
                 train_text,
                 valid_text)

logdir = "HM_LSTM/logging/first_summary_log"
model.run(1,                # number of times learning_rate is decreased
          0.9,              # a factor by which learning_rate is decreased
            200,            # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            50,             # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            3,              # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            1,              # when train point is obtained validation may be performed
            20,             # when train point percent is calculated results got on averaging_number chunks are averaged
          fixed_number_of_steps=5000,
           debug=True,
          save_path="HM_LSTM/new_feature")

"""add_operations=['self.L2_train'],
print_steps=[10, 50, 200],
validation_add_operations = ['self.L2_validation'],
num_validation_prints=10,
print_intermediate_results = True,
summarizing_logdir=logdir"""


