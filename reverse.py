
# coding: utf-8

# In[1]:


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

version = sys.version_info[0]


# In[2]:


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


# In[3]:


#different
offset = 20000
valid_size = 22500
valid_text = text[offset:offset+valid_size]
train_text = text[offset+valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# In[4]:


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


# In[5]:


batch_size_test=64
num_unrollings_test=10

train_batches_test = BatchGenerator(train_text,
                                    batch_size_test,
                                    vocabulary_size,
                                    characters_positions_in_vocabulary,
                                    num_unrollings_test)
valid_batches_test = BatchGenerator(valid_text,
                                    1,
                                    vocabulary_size,
                                    characters_positions_in_vocabulary,
                                    1)

print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))


# In[43]:


indices_GL = {"batch_size": 0,
              "num_unrollings": 1,
              "num_layers": 2,
              "num_nodes": 3,
              "half_life": 4,
              "decay": 5,
              "num_steps": 6,
              "averaging_number": 7,
              "type": 8}


class reverse(MODEL):
    def first_or_middle_layer(self, 
                              inp_t_or_mem_down_t_minus_1,
                              mem_up_t_minus_1,
                              out_or_rem_down_t_minus_1,
                              rem_up_t_minus_1,
                              state_t_minus_1,
                              layer_num):
        X_t = tf.concat([inp_t_or_mem_down_t_minus_1,
                         mem_up_t_minus_1,
                         out_or_rem_down_t_minus_1,
                         rem_up_t_minus_1],
                        1)
        RES = tf.matmul(X_t, self.Matrices[layer_num]) + self.Biases[layer_num]
        i, f, o_or_r_down, r_up, j = tf.split(RES, 5, 1)
        i_gate = tf.sigmoid(i)
        f_gate = tf.sigmoid(f)
        r_up_gate = tf.sigmoid(r_up)
        o_or_r_down_gate = tf.sigmoid(o_or_r_down)
        state_t = f_gate * state_t_minus_1 + i_gate * tf.tanh(j)
        TANH_STATE = tf.tanh(state_t)
        out_or_rem_down_t = o_or_r_down_gate * TANH_STATE
        rem_up_t = r_up_gate * TANH_STATE
        one_layer_gates = tf.concat([i_gate, f_gate, o_or_r_down_gate, r_up_gate], 1)
        return out_or_rem_down_t, state_t, rem_up_t, one_layer_gates
    
    def last_layer(self,
                   mem_down_t_minus_1,
                   rem_down_t_minus_1,
                   state_t_minus_1):
        X_t = tf.concat([mem_down_t_minus_1, rem_down_t_minus_1], 1)
        RES = tf.matmul(X_t, self.Matrices[-1]) + self.Biases[-1]
        i, f, r_down, j = tf.split(RES, 4, axis=1)
        i_gate = tf.sigmoid(i)
        f_gate = tf.sigmoid(f)
        r_down_gate = tf.sigmoid(r_down)  
        state_t = f_gate * state_t_minus_1 + i_gate * tf.tanh(j)
        TANH_STATE = tf.tanh(state_t)
        rem_down_t = r_down_gate * TANH_STATE
        one_layer_gates = tf.concat([i_gate, f_gate, r_down_gate], 1)
        return rem_down_t, state_t, one_layer_gates
    
    def iteration(self, inp, state):
        num_layers = len(state)
        new_state = list()
        gates = list()
        out, inter_state, rem_up, layer_gates = self.first_or_middle_layer(inp,
                                                                           state[1][0],
                                                                           state[0][0],
                                                                           state[0][2],
                                                                           state[0][1],
                                                                           0)
        new_state.append((out, inter_state, rem_up))
        gates.append(layer_gates)
        if num_layers > 2:
            for i in range(num_layers-2):
                rem_down, inter_state, rem_up, layer_gates = self.first_or_middle_layer(state[i][2],
                                                                                        state[i+2][0],
                                                                                        state[i+1][0],
                                                                                        state[i+1][2],
                                                                                        state[i+1][1],
                                                                                        i+1)
                new_state.append((rem_down, inter_state, rem_up))
                gates.append(layer_gates)
        rem_down, inter_state, layer_gates = self.last_layer(state[-2][2],
                                                             state[-1][0],
                                                             state[-1][1])
        new_state.append((rem_down, inter_state))
        gates.append(layer_gates)
        return out, new_state, tf.concat(gates, 1)
    
    def __init__(self,
                 batch_size,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 num_unrollings,
                 num_layers,
                 num_nodes,
                 train_text,
                 valid_text):
        self._results = list()
        self._batch_size = batch_size
        self._vocabulary = vocabulary
        self._vocabulary_size = len(vocabulary)
        self._characters_positions_in_vocabulary = characters_positions_in_vocabulary
        self._num_unrollings = num_unrollings
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._train_text = train_text
        self._valid_text = valid_text
        self._valid_size = len(valid_text)
        self._indices = {"batch_size": 0,
                         "num_unrollings": 1,
                         "num_layers": 2,
                         "num_nodes": 3,
                         "half_life": 4,
                         "decay": 5,
                         "num_steps": 6,
                         "averaging_number": 7,
                         "type": 8}
        self._graph = tf.Graph()
        
        self._last_num_steps = 0
        with self._graph.as_default():
            self._global_step = tf.Variable(0)
            with self._graph.device('/gpu:0'): 
                self.Matrices = list()
                self.Biases = list()
                self.Matrices.append(tf.Variable(tf.truncated_normal([self._vocabulary_size + 2*self._num_nodes[0] + self._num_nodes[1],
                                                                      5 * self._num_nodes[0]],
                                                                     mean=-0.1, stddev=0.1)))
                self.Biases.append(tf.Variable(tf.zeros([5 * self._num_nodes[0]])))
                if self._num_layers > 2:
                    for i in range(self._num_layers - 2):
                        self.Matrices.append(tf.Variable(tf.truncated_normal([self._num_nodes[i] + 2*self._num_nodes[i+1] + self._num_nodes[i+2],
                                                                              5 * self._num_nodes[i+1]],
                                                                             mean=-0.1, stddev=0.1)))
                        self.Biases.append(tf.Variable(tf.zeros([5 * self._num_nodes[i+1]])))
                self.Matrices.append(tf.Variable(tf.truncated_normal([self._num_nodes[-1] + self._num_nodes[-2],
                                                                      4 * self._num_nodes[-1]],
                                                                     mean=-0.1, stddev=0.1)))     
                self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[-1]])))


                # classifier 
                weights = tf.Variable(tf.truncated_normal([self._num_nodes[0], self._vocabulary_size], stddev = 0.1))
                bias = tf.Variable(tf.zeros([self._vocabulary_size]))
                
                """PLACEHOLDERS train data"""
                self._train_data = list()
                for _ in range(self._num_unrollings + 1):
                    self._train_data.append(
                        tf.placeholder(tf.float32, shape=[self._batch_size, self._vocabulary_size]))
                train_inputs = self._train_data[: self._num_unrollings]
                train_labels = self._train_data[1:]  # labels are inputs shifted by one time step.
                # Unrolled LSTM loop.

                saved_state = list()
                for i in range(self._num_layers-1):
                    saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]]), trainable=False),
                                        tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]]), trainable=False),
                                        tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]]), trainable=False)))
                saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]]), trainable=False),
                                    tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]]), trainable=False)))

                outputs = list()
                state = saved_state
                for i in train_inputs:
                    output, state, _ = self.iteration(i, state)
                    outputs.append(output)

                save_list = list()
                for i in range(self._num_layers-1):
                    save_list.append(saved_state[i][0].assign(state[i][0]))
                    save_list.append(saved_state[i][1].assign(state[i][1]))
                    save_list.append(saved_state[i][2].assign(state[i][2]))
                save_list.append(saved_state[-1][0].assign(state[-1][0]))
                save_list.append(saved_state[-1][1].assign(state[-1][1]))
                
                """skip operation"""
                self._skip_operation = tf.group(*save_list)

                with tf.control_dependencies(save_list):
                        # Classifier.
                    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), weights, bias)
                    """loss"""
                    self._loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.concat(train_labels, 0), logits=logits))
                # Optimizer.
                
                """PLACEHOLDERS half life and decay"""
                self._half_life = tf.placeholder(tf.int32)
                self._decay = tf.placeholder(tf.float32)
                """learning rate"""
                self._learning_rate = tf.train.exponential_decay(10.0,
                                                                 self._global_step,
                                                                 self._half_life,
                                                                 self._decay,
                                                                 staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(self._loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
                """optimizer"""
                self._optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=self._global_step)
                """train prediction"""
                self._train_prediction = tf.nn.softmax(logits)

                # Sampling and validation eval: batch 1, no unrolling.
                saved_sample_state = list()
                for i in range(self._num_layers-1):
                    saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[i]]), trainable=False),
                                               tf.Variable(tf.zeros([1, self._num_nodes[i]]), trainable=False),
                                               tf.Variable(tf.zeros([1, self._num_nodes[i]]), trainable=False)))
                saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[-1]]), trainable=False),
                                           tf.Variable(tf.zeros([1, self._num_nodes[-1]]), trainable=False))) 
                """PLACEHOLDER sample input"""
                self._sample_input = tf.placeholder(tf.float32, shape=[1, self._vocabulary_size])

                reset_list = list()
                for i in range(self._num_layers-1):
                    reset_list.append(saved_sample_state[i][0].assign(tf.zeros([1, self._num_nodes[i]])))
                    reset_list.append(saved_sample_state[i][1].assign(tf.zeros([1, self._num_nodes[i]])))
                    reset_list.append(saved_sample_state[i][2].assign(tf.zeros([1, self._num_nodes[i]])))
                reset_list.append(saved_sample_state[-1][0].assign(tf.zeros([1, self._num_nodes[-1]])))
                reset_list.append(saved_sample_state[-1][1].assign(tf.zeros([1, self._num_nodes[-1]])))
                """reset sample state"""
                self._reset_sample_state = tf.group(*reset_list)

                sample_output, sample_state, self.gates = self.iteration(self._sample_input, saved_sample_state)

                sample_save_list = list()
                for i in range(self._num_layers-1):
                    sample_save_list.append(saved_sample_state[i][0].assign(sample_state[i][0]))
                    sample_save_list.append(saved_sample_state[i][1].assign(sample_state[i][1])) 
                    sample_save_list.append(saved_sample_state[i][2].assign(sample_state[i][2])) 
                sample_save_list.append(saved_sample_state[-1][0].assign(sample_state[-1][0]))
                sample_save_list.append(saved_sample_state[-1][1].assign(sample_state[-1][1]))

                with tf.control_dependencies(sample_save_list):
                    """sample prediction"""
                    self._sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, weights, bias)) 
                
                """Matrices = list()
                Biases = list()
                gates = list()
                with tf.variable_scope(LSTM_scope, reuse=True):
                    for i in range(self._num_layers):
                        with tf.variable_scope("Cell%d" % i):
                            with tf.variable_scope("BasicLSTMCell/Linear"):
                                Matrices.append(tf.get_variable("Matrix"))
                                Biases.append(tf.get_variable("Bias"))
                cur_inp = self._sample_input
                for layer_num in range(self._num_layers):
                    concat = tf.matmul(tf.concat(1, [cur_inp,
                                                     saved_sample_state[layer_num][1]]),
                                       Matrices[layer_num]) + Biases[layer_num]
                    i, _, f, o = tf.split(1, 4, concat)
                    gates.append(tf.concat(0,
                                           [tf.sigmoid(i),
                                            tf.sigmoid(f),
                                            tf.sigmoid(o)]))
                    with tf.variable_scope(LSTM_scope, reuse=True):
                        with tf.variable_scope("Cell%d" % layer_num):
                            cur_inp, _ = cell_list[layer_num](cur_inp, saved_sample_state[layer_num])
                    
                #gates
                self.gates = tf.pack(gates)"""
                
                """saver"""
            self.saver = tf.train.Saver(max_to_keep=None)
                            
                        
    
    def _generate_metadata(self, half_life, decay, num_averaging_iterations):
        metadata = list()
        metadata.append(self._batch_size)
        metadata.append(self._num_unrollings)
        metadata.append(self._num_layers)
        metadata.append(self._num_nodes)
        metadata.append(half_life)
        metadata.append(decay)
        metadata.append(self._last_num_steps)
        metadata.append(num_averaging_iterations)
        metadata.append('reverse')
        return metadata
        
        
    def get_gates(self, session, num_strings=10, length=75, start_positions=None):
        self._reset_sample_state.run()
        self._valid_batches = BatchGenerator(self._valid_text,
                                             1,
                                             self._vocabulary_size,
                                             self._characters_positions_in_vocabulary,
                                             1)
        if start_positions is None:
            start_positions = list()
            if self._valid_size / num_strings < length:
                num_strings = self._valid_size / length
            for i in range(num_strings):
                start_positions.append(i* (self._valid_size / num_strings) + self._valid_size / num_strings / 2)
            while self._valid_size - start_positions[-1] < length:
                del start_positions[-1]
        text_list = list()
        gate_dict = dict()
        aver_gate_dict = dict()
        first_order = ['i', 'f', 'd', 'u']
        second_order = ['i', 'f', 'd']
        for layer_idx in range(self._num_layers-1):
            gate_dict[layer_idx+1] = {'i': list(), 'f': list(), 'd': list(), 'u': list()}
            aver_gate_dict[layer_idx+1] = {'i': list(), 'f': list(), 'd': list(), 'u': list()}
        gate_dict[self._num_layers] = {'i': list(), 'f': list(), 'd': list()}
        aver_gate_dict[self._num_layers] = {'i': list(), 'f': list(), 'd': list()}
        collect_gates = False
        letters_parsed = -1
        for idx in range(self._valid_size):
            b = self._valid_batches.next()
            
            if idx in start_positions or collect_gates: 
                if letters_parsed == -1:
                    letters_parsed = 0
                    text = u""
                    g_dict = dict()
                    for layer_idx in range(self._num_layers-1):
                        g_dict[layer_idx+1] = {'i': [list() for _ in range(self._num_nodes[layer_idx])],
                                               'f': [list() for _ in range(self._num_nodes[layer_idx])],
                                               'd': [list() for _ in range(self._num_nodes[layer_idx])],
                                               'u': [list() for _ in range(self._num_nodes[layer_idx])]}
                    g_dict[self._num_layers] = {'i': [list() for _ in range(self._num_nodes[self._num_layers-1])],
                                                  'f': [list() for _ in range(self._num_nodes[self._num_layers-1])],
                                                  'd': [list() for _ in range(self._num_nodes[self._num_layers-1])]}
                    av_g_dict = dict()
                    for layer_idx in range(self._num_layers-1):
                        av_g_dict[layer_idx+1] = {'i': list(),
                                                  'f': list(),
                                                  'd': list(),
                                                  'u': list()}
                    av_g_dict[self._num_layers] = {'i': list(),
                                                     'f': list(),
                                                     'd': list()}
                    collect_gates = True
                text += characters(b[0], self._vocabulary)[0]
                iteration_gates = self.gates.eval({self._sample_input: b[0]})
                num_gates = iteration_gates.shape[0]
                #print('num_gates: ', num_gates)
                gates_by_layer = list()
                av_gates_by_layer = list()
                start = 0
                for layer_idx in range(self._num_layers-1):
                    stop = start + self._num_nodes[layer_idx] * 4
                    layer_gates = iteration_gates[:, start : stop]
                    layer_gates_list = list()
                    av_layer_gates_list = list()
                    for gate_idx in range(4):
                        layer_gates_list.append(
                            list(
                                np.squeeze(
                                    layer_gates[:, gate_idx * self._num_nodes[layer_idx] : (gate_idx+1) * self._num_nodes[layer_idx]])))
                        av_layer_gates_list.append(
                            np.mean(
                                layer_gates[:, gate_idx * self._num_nodes[layer_idx] : (gate_idx+1) * self._num_nodes[layer_idx]]))
                    gates_by_layer.append(layer_gates_list)
                    av_gates_by_layer.append(av_layer_gates_list)
                    start += self._num_nodes[layer_idx]

                layer_gates = iteration_gates[:, start : ]
                layer_gates_list = list()
                av_layer_gates_list = list()
                for gate_idx in range(3):
                    layer_gates_list.append(
                        list(
                            np.squeeze(
                                layer_gates[:, gate_idx * self._num_nodes[layer_idx] : (gate_idx+1) * self._num_nodes[layer_idx]])))
                    av_layer_gates_list.append(
                        np.mean(
                            layer_gates[:, gate_idx * self._num_nodes[layer_idx] : (gate_idx+1) * self._num_nodes[layer_idx]]))
                gates_by_layer.append(layer_gates_list)
                av_gates_by_layer.append(av_layer_gates_list)
                
                for layer_idx in range(self._num_layers-1):
                    #print(av_g_dict[layer_idx+1].keys())
                    for gate_idx, gate_type in enumerate(first_order):
                        av_g_dict[layer_idx+1][gate_type].append(av_gates_by_layer[layer_idx][gate_idx])
                        #print(len(gates_by_layer[layer_idx][gate_idx]))
                        for node_idx, node_value in enumerate(gates_by_layer[layer_idx][gate_idx]):
                            #print(node_idx)
                            g_dict[layer_idx+1][gate_type][node_idx].append(node_value)
                for gate_idx, gate_type in enumerate(second_order):
                    av_g_dict[self._num_layers][gate_type].append(av_gates_by_layer[self._num_layers-1][gate_idx])
                    for node_idx, node_value in enumerate(gates_by_layer[self._num_layers-1][gate_idx]):
                        g_dict[self._num_layers][gate_type][node_idx].append(node_value)
                letters_parsed += 1
                if letters_parsed >= length:
                    collect_gates = False
                    for layer_key in gate_dict.keys():
                        for gate_key in gate_dict[layer_key].keys():
                            gate_dict[layer_key][gate_key].append(g_dict[layer_key][gate_key])
                            aver_gate_dict[layer_key][gate_key].append(av_g_dict[layer_key][gate_key])
                    text_list.append(text)
                    letters_parsed = -1
                    
            _ = self._sample_prediction.eval({self._sample_input: b[0]})
        return text_list, gate_dict, aver_gate_dict    


# In[44]:


model = reverse(64,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 50,
                 3,
                 [1000, 1000, 1000],
                 train_text,
                 valid_text)






model.run(1,                # number of times learning_rate is decreased
          0.9,              # a factor by which learning_rate is decreased
            200,            # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            50,             # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            3,              # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            1,              # when train point is obtained validation may be performed
            20,             # when train point percent is calculated results got on averaging_number chunks are averaged
          fixed_number_of_steps=5000,
          save_path="reverse_LSTM/test_for_nan")
