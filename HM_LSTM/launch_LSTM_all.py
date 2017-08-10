import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs
import time
import os
import gc
from six.moves import cPickle as pickle
import sys
import subprocess
if not os.path.isfile('model_module.py') or not os.path.isfile('plot_module.py'):
    current_path = os.path.dirname(os.path.abspath('__file__'))
    additional_path = '/'.join(current_path.split('/')[:-1])
    sys.path.append(additional_path)
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


from LSTM_all_core import LSTM

version = sys.version_info[0]

class CommandLineInput(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

if len(sys.argv) < 2:
    raise CommandLineInput("2nd command line argument indicating dataset type is missing.\nUse either 'clean' or 'dirty'")
if sys.argv[1] == 'dirty':
    if not os.path.exists('enwik8_filtered'):
        if not os.path.exists('enwik8'):
            filename = maybe_download('enwik8.zip', 36445475)
            full_text = read_data(filename)
            f = open('enwik8', 'wb')
            f.write(full_text.encode('utf8'))
            f.close()
        else:
            f = open('enwik8', 'rb')
            full_text = f.read().decode('utf8')
            f.close()
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
elif sys.argv[1] == 'clean':
    if not os.path.exists('enwik8_clean'):
        if not os.path.exists('enwik8'):
            filename = maybe_download('enwik8.zip', 36445475)
            full_text = read_data(filename)
            f = open('enwik8', 'wb')
            f.write(full_text.encode('utf8'))
            f.close()       
        perl_script = subprocess.call(['perl', "clean.pl", 'enwik8', 'enwik8_clean'])
    f = open('enwik8_clean', 'rb')
    text = f.read().decode('utf8')
    print(len(text))
    f.close() 
    (not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters_indices) = check_not_one_byte(text)

else:
    raise CommandLineInput("2nd command line argument indicating dataset type is wrong.\nUse either 'clean' or 'dirty'")


#different
offset = 20000
valid_size = 10000
valid_text = text[offset:offset+valid_size]
train_text = text[offset+valid_size:]
train_size = len(train_text)


# In[5]:


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

print('vocabulary_size: ', vocabulary_size)
string_vocabulary = u""
for i in range(vocabulary_size):
    string_vocabulary += vocabulary[i]


model = LSTM(64,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 100,
                 3,
                 [256, 256, 256],
                 train_text,
                 valid_text,
                init_parameter=1e-6,
                 matr_init_parameter=100000)


model.run(20,                # number of times learning_rate is decreased
          0.9,              # a factor by which learning_rate is decreased
            100,            # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            500,             # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            3,              # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            1,              # when train point is obtained validation may be performed
            3,             # when train point percent is calculated results got on averaging_number chunks are averaged
          fixed_number_of_steps=50001,
          validation_example_length=40, 
           #debug=True,
            print_intermediate_results = True,
          path_to_file_for_saving_prints='peganov/LSTM_all/effectiveness_clean/effectiveness_clean.txt',
           save_path="peganov/LSTM_all/effectiveness_clean/variables",
           gpu_memory=0.4)
results_GL = list(model._results)

folder_name = 'peganov/LSTM_all/effectiveness_clean'
file_name = 'effectiveness_clean_result.pickle'
force = True
pickle_dump = {'results_GL': results_GL}
if not os.path.exists(folder_name):
    try:
        os.makedirs(folder_name)
    except Exception as e:
        print("Unable create folder '%s'" % folder_name, ':', e)    
print('Pickling %s' % (folder_name + '/' + file_name))
try:
    with open(folder_name + '/' + file_name, 'wb') as f:
        pickle.dump(pickle_dump, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', file_name, ':', e)
