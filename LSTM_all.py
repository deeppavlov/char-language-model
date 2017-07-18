
# coding: utf-8

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import math
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
 


# In[3]:


#different
offset_1 = 0
offset_2 = 4100
valid_size_1 = 4000
valid_size_2 = 4000
valid_text_1 = text[offset_1:offset_1+valid_size_1]
valid_text_2 = text[offset_2:offset_2+valid_size_2]
train_text = text[offset_2+valid_size_2:]
train_size = len(train_text)


# In[4]:


#different
offset = 20000
valid_size = 25000
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


string_vocabulary = u""
for i in range(vocabulary_size):
    string_vocabulary += vocabulary[i]


# In[6]:


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


# In[19]:


# This class implements hierarchical LSTM described in the paper https://arxiv.org/pdf/1609.01704.pdf
# All variables names and formula indices are taken from mentioned article
# notation A^i stands for A with upper index i
# notation A_i stands for A with lower index i
# notation A^i_j stands for A with upper index i and lower index j
class LSTM(MODEL):
        
    def L2_norm(self,
                tensor,
                dim):
        with tf.name_scope('L2_norm'):
            square = tf.square(tensor, name="square_in_L2_norm")
            reduced = tf.reduce_mean(square,
                                     dim,
                                     keep_dims=True,
                                     name="reduce_mean_in_L2_norm")
            return tf.sqrt(reduced, name="L2_norm")
    
    
    def layer(self,
              idx,                   
              state,                 
              bottom_up):

        with tf.name_scope('LSTM_layer_%s'%(idx)):
            # batch_size of processed data
            current_batch_size = bottom_up.get_shape().as_list()[0]

            X = tf.concat([bottom_up, state[0]],
                          1,
                          name="X")
            concat = tf.add(tf.matmul(X,
                                      self.Matrices[idx],
                                      name="matmul_in_concat"),
                            self.Biases[idx],
                            name="concat")

            # following operations implement function vector implementation in formula (4)
            # and compute f^l_t, i^l_t, o^l_t, g^l_t and z^l_t
            [sigmoid_arg, tanh_arg] = tf.split(concat,
                                               [3*self._num_nodes[idx], self._num_nodes[idx]],
                                               axis=1,
                                               name="split_to_function_arguments")

            
            gate_concat = tf.sigmoid(sigmoid_arg, name="gate_concat")
            [forget_gate, input_gate, output_gate] = tf.split(gate_concat,
                                                              3,
                                                              axis=1,
                                                              name="split_to_gates_op")
            modification_vector = tf.tanh(tanh_arg, name="modification_vector")

            prepaired_input = tf.multiply(modification_vector, input_gate, name="prepaired_input")
            prepaired_memory = tf.multiply(state[1], output_gate, name="prepaired_memory")
            new_memory = tf.add(prepaired_memory, prepaired_input, name="new_memory")
            new_hidden = tf.multiply(tf.tanh(new_memory, name="tanh_result_for_new_hidden"),
                                     output_gate,
                                     name="new_hidden")
        return new_hidden, new_memory
    

    
    def iteration(self, inp, state, iter_idx):
        # This function implements processing of one character embedding by HM_LSTM
        # 'inp' is one character embedding
        # 'state' is network state from previous layer
        # Method returns: new state of the network which includes hidden states,
        # memory states and boundary states for all layers; concatenated boundaries for all
        # layers ([batch_size, self._num_layers-1])
        
        with tf.name_scope('iteration_%s'%iter_idx):
            hidden = inp
            new_state = list()
            for idx in range(self._num_layers):
                hidden, memory = self.layer(idx,
                                            state[idx],
                                            hidden)
                new_state.append((hidden, memory))

            return new_state
    
    def embedding_module(self,
                         inputs):
        # This function embeds input one-hot encodded character vector into a vector of dimension embedding_size
        # For computation acceleration inputs are concatenated before being multiplied on self.embedding_weights
        with tf.name_scope('embedding_module'):
            current_num_unrollings = len(inputs)
            inputs = tf.concat(inputs,
                               0,
                               name="inputs_concat_in_embedding_module")
            embeddings = tf.matmul(inputs,
                                   self.embedding_weights,
                                   name="concatenated_embeddings_in_embedding_module")
            return tf.split(embeddings,
                            current_num_unrollings,
                            axis=0,
                            name="embedding_module_output")
    
    
    def RNN_module(self,
                   embedded_inputs,
                   saved_state):

        with tf.name_scope('RNN_module'):
            # 'saved_hidden_states' is a list of self._num_layers elements. idx-th element of the list is 
            # a concatenation of hidden states on idx-th layer along chunk of input text.
            saved_hidden_states = list()
            for _ in range(self._num_layers):
                saved_hidden_states.append(list())
            state = saved_state
            for emb_idx, emb in enumerate(embedded_inputs):
                state  = self.iteration(emb, state, emb_idx)
                for layer_state, saved_hidden in zip(state, saved_hidden_states):
                    saved_hidden.append(layer_state[0])
            for idx, layer_saved in enumerate(saved_hidden_states):
                saved_hidden_states[idx] = tf.concat(layer_saved, 0, name="hidden_concat_in_RNN_module_on_layer%s"%idx)

            return state, saved_hidden_states
            
    
    def output_module(self,
                      hidden_states):
        with tf.name_scope('output_module'):
            concat = tf.concat(hidden_states, 1, name="total_concat_hidden")
            output_module_gates = tf.transpose(tf.sigmoid(tf.matmul(concat,
                                                                    self.output_module_gates_weights,
                                                                    name="matmul_in_output_module_gates"),
                                                          name="sigmoid_in_output_module_gates"),
                                               name="output_module_gates")
            output_module_gates = tf.split(output_module_gates,
                                           self._num_layers,
                                           axis=0,
                                           name="split_of_output_module_gates")
            tr_gated_hidden_states = list()
            for idx, hidden_state in enumerate(hidden_states):
                tr_hidden_state = tf.transpose(hidden_state, name="tr_hidden_state_total_%s"%idx)
                tr_gated_hidden_states.append(tf.multiply(output_module_gates[idx],
                                                          tr_hidden_state,
                                                          name="tr_gated_hidden_states_%s"%idx))
            gated_hidden_states = tf.transpose(tf.concat(tr_gated_hidden_states,
                                                         0,
                                                         name="concat_in_gated_hidden_states"),
                                               name="gated_hidden_states")
            return tf.add(tf.matmul(gated_hidden_states,
                                    self.output_weights,
                                    name="matmul_in_logits_output"),
                          self.output_bias,
                          name="logits")
        
    
    def __init__(self,
                 batch_size,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 num_unrollings,
                 num_layers,
                 num_nodes,
                 train_text,
                 valid_text,
                 embedding_size=128):
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
        self._embedding_size = embedding_size
        self._indices = {"batch_size": 0,
                         "num_unrollings": 1,
                         "num_layers": 2,
                         "num_nodes": 3,
                         "half_life": 4,
                         "decay": 5,
                         "num_steps": 6,
                         "averaging_number": 7,
                         "embedding_size": 11,
                         "type": 12}
        self._graph = tf.Graph()
        
        self._last_num_steps = 0
        with self._graph.as_default(): 
            with tf.name_scope('train'):
                self._global_step = tf.Variable(0, trainable=False, name="global_step")
            with self._graph.device('/gpu:0'):
                # embedding module variables
                self.embedding_weights = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._embedding_size],
                                                                         stddev = math.sqrt(1./self._vocabulary_size),
                                                                         name="embeddings_matrix_initialize"), 
                                                     name="embeddings_matrix_variable")
                
                # RNN module variables
                self.Matrices = list()
                self.Biases = list()
                
                # tensor name templates for HM_LSTM parameters
                init_matr_name = "HM_LSTM_matrix_%s_initializer"
                init_bias_name = "HM_LSTM_bias_%s_initializer" 
                matr_name = "HM_LSTM_matrix_%s"
                bias_name = "HM_LSTM_bias_%s"
                
                self.Matrices.append(tf.Variable(tf.truncated_normal([self._embedding_size + self._num_nodes[0],
                                                                      4 * self._num_nodes[0]],
                                                                     mean=0.,
                                                                     stddev=math.sqrt(1./(self._embedding_size+self._num_nodes[0])),
                                                                     name=init_matr_name%0),
                                                 name=matr_name%0))
                self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[0]],
                                                        name=init_bias_name%0),
                                               name=bias_name%0))
                if self._num_layers > 1:
                    for i in range(self._num_layers - 1):
                        self.Matrices.append(tf.Variable(tf.truncated_normal([self._num_nodes[i] + self._num_nodes[i+1],
                                                                              4 * self._num_nodes[i+1]],
                                                                             mean=0.,
                                                                             stddev=math.sqrt(1./(self._num_nodes[i]+self._num_nodes[i+1])),
                                                                             name=init_matr_name%(i+1)),
                                                         name=matr_name%(i+1)))
                        self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[i+1]],
                                                                name=init_bias_name%(i+1)),
                                                       name=bias_name%(i+1)))

                dim_classifier_input = sum(self._num_nodes)
                
                # output module variables
                # output module gates weights (w^l vectors in (formula (11)))
                self.output_module_gates_weights = tf.Variable(tf.truncated_normal([dim_classifier_input, self._num_layers],
                                                                                   stddev = math.sqrt(1./dim_classifier_input),
                                                                                   name="output_gates_weights_initializer"),
                                                               name="output_gates_weights")
                # classifier 
                self.output_weights = tf.Variable(tf.truncated_normal([dim_classifier_input, self._vocabulary_size],
                                                                      stddev = math.sqrt(1./dim_classifier_input),
                                                                      name="output_weights_initializer"),
                                                  name="output_weights")
                self.output_bias = tf.Variable(tf.zeros([self._vocabulary_size], name="output_bias_initializer"),
                                               name="output_bias")
                
                
                with tf.name_scope('train'):
                    """PLACEHOLDERS train data"""
                    # data input placeholder name template
                    inp_name_templ = "placeholder_inp_%s"
                    self._train_data = list()
                    for j in range(self._num_unrollings + 1):
                        self._train_data.append(
                            tf.placeholder(tf.float32,
                                           shape=[self._batch_size, self._vocabulary_size],
                                           name=inp_name_templ%j))
                    train_inputs = self._train_data[: self._num_unrollings]
                    train_labels = self._train_data[1:]  # labels are inputs shifted by one time step.
                    # Unrolled LSTM loop.


                    saved_state = list()
                    # templates for saved_state tensor names
                    saved_state_init_templ = "saved_state_layer%s_number%s_initializer"
                    saved_state_templ = "saved_state_layer%s_number%s"
                    for i in range(self._num_layers):
                        saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]],
                                                                 name=saved_state_init_templ%(i, 0)),
                                                        trainable=False,
                                                        name=saved_state_templ%(i, 0)),
                                            tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]],
                                                                 name=saved_state_init_templ%(i, 1)),
                                                        trainable=False,
                                                        name=saved_state_templ%(i, 1))))


                    embedded_inputs = self.embedding_module(train_inputs)
                    state, hidden_states = self.RNN_module(embedded_inputs, saved_state)
                    logits = self.output_module(hidden_states)

                    save_list = list()
                    save_list_templ = "save_list_assign_layer%s_number%s"
                    for i in range(self._num_layers):
                        save_list.append(tf.assign(saved_state[i][0],
                                                   state[i][0],
                                                   name=save_list_templ%(i, 0)))
                        save_list.append(tf.assign(saved_state[i][1],
                                                   state[i][1],
                                                   name=save_list_templ%(i, 1)))


                    """skip operation"""
                    self._skip_operation = tf.group(*save_list, name="skip_operation")

                    with tf.control_dependencies(save_list):
                            # Classifier.
                        """loss"""
                        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(train_labels,
                                                                                                             0,
                                                                                                             name="train_labels_concat_for_cross_entropy"),
                                                                                            logits=logits,
                                                                                            name="cross_entropy"),
                                                    name="reduce_mean_for_loss_computation")
                    # Optimizer.

                    # global variables initializer
                    self.global_initializer = tf.global_variables_initializer()

                    """PLACEHOLDERS half life and decay"""
                    self._half_life = tf.placeholder(tf.int32, name="half_life")
                    self._decay = tf.placeholder(tf.float32, name="decay")
                    """learning rate"""
                    
                    # A list of first dimensions of all matrices
                    # It is used for defining initial learning rate
                    dimensions = list()
                    dimensions.append(self._vocabulary_size)
                    dimensions.append(self._embedding_size + self._num_nodes[0] + self._num_nodes[1])
                    if self._num_layers > 2:
                        for i in range(self._num_layers-2):
                            dimensions.append(self._num_nodes[i] + self._num_nodes[i+1] + self._num_nodes[i+2])
                    dimensions.append(sum(self._num_nodes))
                    max_dimension = max(dimensions)
                    
                    self._learning_rate = tf.train.exponential_decay(160./math.sqrt(max_dimension),
                                                                     self._global_step,
                                                                     self._half_life,
                                                                     self._decay,
                                                                     staircase=True,
                                                                     name="learning_rate")
                    optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
                    gradients, v = zip(*optimizer.compute_gradients(self._loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
                    """optimizer"""
                    self._optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=self._global_step)
                    """train prediction"""
                    self._train_prediction = tf.nn.softmax(logits, name="train_prediction")

                # Sampling and validation eval: batch 1, no unrolling.
                with tf.name_scope('validation'):
                    saved_sample_state = list()
                    saved_state_init_templ = "saved_sample_state_layer%s_number%s_initializer"
                    saved_state_templ = "saved_sample_state_layer%s_number%s"
                    for i in range(self._num_layers):
                        saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[i]],
                                                                        name=saved_state_init_templ%(i, 0)),
                                                               trainable=False,
                                                               name=saved_state_templ%(i, 0)),
                                                   tf.Variable(tf.zeros([1, self._num_nodes[i]],
                                                                        name=saved_state_init_templ%(i, 1)),
                                                               trainable=False,
                                                               name=saved_state_templ%(i, 1))))


                    """PLACEHOLDER sample input"""
                    self._sample_input = tf.placeholder(tf.float32,
                                                        shape=[1, self._vocabulary_size],
                                                        name="sample_input_placeholder")

                    reset_list_templ = "reset_list_assign_layer%s_number%s"
                    saved_state_init_templ = "saved_state_layer%s_number%s_initializer"
                    reset_list = list()
                    for i in range(self._num_layers):
                        reset_list.append(tf.assign(saved_sample_state[i][0],
                                                    tf.zeros([1, self._num_nodes[i]],
                                                             name=saved_state_init_templ%(i, 0)),
                                                    name=reset_list_templ%(i, 0)))
                        reset_list.append(tf.assign(saved_sample_state[i][1],
                                                    tf.zeros([1, self._num_nodes[i]],
                                                             name=saved_state_init_templ%(i, 1)),
                                                    name=reset_list_templ%(i, 1)))
                    #reset sample state
                    self._reset_sample_state = tf.group(*reset_list, name="reset_sample_state")
 

                    sample_embedded_inputs = self.embedding_module([self._sample_input])
                    sample_state, sample_hidden_states = self.RNN_module(sample_embedded_inputs,
                                                                                            saved_sample_state)
                    sample_logits = self.output_module(sample_hidden_states) 

                    sample_save_list = list()
                    save_list_templ = "save_sample_list_assign_layer%s_number%s"
                    for i in range(self._num_layers):
                        sample_save_list.append(tf.assign(saved_sample_state[i][0],
                                                          sample_state[i][0],
                                                          name=save_list_templ%(i, 0)))
                        sample_save_list.append(tf.assign(saved_sample_state[i][1],
                                                          sample_state[i][1],
                                                          name=save_list_templ%(i, 1)))

                    with tf.control_dependencies(sample_save_list):
                        """sample prediction"""
                        self._sample_prediction = tf.nn.softmax(sample_logits, name="sample_prediction") 

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
        metadata.append(self._init_slope)
        metadata.append(self._slope_growth)
        metadata.append(self._slope_half_life)
        metadata.append(self._embedding_size)
        metadata.append('HM_LSTM')
        return metadata
  
    def get_boundaries(self, session, num_strings=10, length=75, start_positions=None):
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
        boundaries_list = list()
        collect_boundaries = False
        letters_parsed = -1
        for idx in range(self._valid_size):
            b = self._valid_batches.next()
            
            if idx in start_positions or collect_boundaries: 
                if letters_parsed == -1:
                    letters_parsed = 0
                    text = u""
                    b_double_list = list()
                    for _ in range(self._num_layers-1):
                        b_double_list.append(list())
                    collect_boundaries = True
                text += characters(b[0], self._vocabulary)[0]
                letter_boundaries = self.boundary.eval({self._sample_input: b[0]})
                for layer_idx, layer_boundaries in enumerate(b_double_list):
                    layer_boundaries.append(letter_boundaries[layer_idx])
                letters_parsed += 1
                if letters_parsed >= length:
                    collect_boundaries = False
                    boundaries_list.append(b_double_list)
                    text_list.append(text)
                    letters_parsed = -1
                    
            _ = self._sample_prediction.eval({self._sample_input: b[0]})
        return text_list, boundaries_list 



model = HM_LSTM(53,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 100,
                 3,
                 [512, 512, 512],
                 train_text,
                 valid_text)

model.run(20,                # number of times learning_rate is decreased
          0.9,              # a factor by which learning_rate is decreased
            1000,            # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            100,             # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            3,              # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            1,              # when train point is obtained validation may be performed
            20,             # when train point percent is calculated results got on averaging_number chunks are averaged
          fixed_number_of_steps=100000,
            print_intermediate_results = True,
          save_path="peganov/LSTM_all/first/variables")

