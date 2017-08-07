
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


# In[4]:


#different
offset = 20000
valid_size = 25000
valid_text = text[offset:offset+valid_size]
train_text = text[offset+valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


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

print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(train_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))
print(batches2string(valid_batches_test.next(), vocabulary))


# In[7]:


# This class implements hierarchical LSTM described in the paper https://arxiv.org/pdf/1609.01704.pdf
# All variables names and formula indices are taken from mentioned article
# notation A^i stands for A with upper index i
# notation A_i stands for A with lower index i
# notation A^i_j stands for A with upper index i and lower index j
class HM_LSTM(MODEL):
    
        
    def L2_norm(self,
                tensor,
                dim,
                appendix):
        with tf.name_scope('L2_norm'):
            square = tf.square(tensor, name="square_in_L2_norm"+appendix)
            reduced = tf.reduce_mean(square,
                                     dim,
                                     keep_dims=True,
                                     name="reduce_mean_in_L2_norm"+appendix)
            return tf.sqrt(reduced, name="L2_norm"+appendix)
    
    def step_function(self,
                      inp_tensor,
                      appendix):
        sign_res = tf.sign(inp_tensor, name="sign_in_step_function"+appendix)
        add_res = tf.add(sign_res, 1., name="add_in_step_function"+appendix)
        return tf.divide(add_res, 2., name="step_func"+appendix)
    
    def not_last_layer(self,
                       idx,                   # layer number (from 0 to self._num_layers - 1)
                       emb_idx,
                       state,                 # A tuple of tensors containing h^l_{t-1}, c^l_{t-1} and z^l_{t-1}
                       bottom_up,             # A tensor h^{l-1}_t  
                       top_down,              # A tensor h^{l+1}_{t-1}
                       boundary_state_down,   # A tensor z^{l-1}_t
                       appendix):
        # method implements operations (2) - (7) (shortly (1)) performed on idx-th layer
        # ONLY NOT FOR LAST LAYER! Last layer computations are implemented in self.last_layer method
        # and returns 3 tensors: hidden state, memory state
        # and boundary state (h^l_t, c^l_t and z^l_t accordingly)
        
        with tf.name_scope('LSTM_layer_%s'%(idx)):
            # batch_size of processed data
            current_batch_size = bottom_up.get_shape().as_list()[0]


            # note: in several next operations tf.transpose method is applied repeatedly.
            #       It was used for broadcasting activation along vectors in same batches

            # following operation computes a product z^l_{t-1} x h^{l+1}_{t-1} for formula (6)
            top_down_prepaired = tf.transpose(tf.multiply(tf.transpose(state[2],
                                                                       name="transposed_state2_in_top_down_prepaired"+appendix),
                                                          tf.transpose(top_down,
                                                                       name="transposed_top_down_in_top_down_prepaired"+appendix),
                                                          name="multiply_in_top_down_prepaired"+appendix),
                                              name="top_down_prepaired"+appendix)

            # this one cumputes a product z^{l-1}_t x h^{l-1}_t for formula (7)
            bottom_up_prepaired = tf.transpose(tf.multiply(tf.transpose(boundary_state_down,
                                                                        name="transposed_boundary_state_down_in_bottom_down_prepaired"+appendix),
                                                           tf.transpose(bottom_up,
                                                                        name="transposed_bottom_up_in_bottom_up_prepaired"+appendix),
                                                           name="multiply_in_bottom_up_prepaired"+appendix),
                                               name="bottom_up_prepaired"+appendix)

            # Matrix multiplications in formulas (5) - (7) and sum in argument of function f_slice
            # in formula (4) are united in one operation
            # Matrices U^l_l, U^l_{l+1} and W^l_{l-1} are concatenated into one matrix self.Matrices[idx]
            # and vectors h^l_{t-1}, z^l_{t-1} x h^{l+1}_{t-1} and  z^{l-1}_t x h^{l-1}_t are 
            # concatenated into vector X
            X = tf.concat([bottom_up_prepaired, state[0], top_down_prepaired],
                          1,
                          name="X"+appendix)
            concat = tf.add(tf.matmul(X,
                                      self.Matrices[idx].read_value(),
                                      name="matmul_in_concat"+appendix),
                            self.Biases[idx].read_value(),
                            name="concat"+appendix)

            # following operations implement function vector implementation in formula (4)
            # and compute f^l_t, i^l_t, o^l_t, g^l_t and z^l_t
            [sigmoid_arg, tanh_arg, hard_sigm_arg] = tf.split(concat,
                                                              [3*self._num_nodes[idx], self._num_nodes[idx], 1],
                                                              axis=1,
                                                              name="split_to_function_arguments"+appendix)
            
            L2_norm_of_hard_sigm_arg = self.L2_norm(hard_sigm_arg,
                                                    0,
                                                    "_hard_sigm"+appendix)
            
            gate_concat = self.step_function(sigmoid_arg, "_gate_concat"+appendix)
            [forget_gate, input_gate, output_gate] = tf.split(gate_concat,
                                                              3,
                                                              axis=1,
                                                              name="split_to_gates_op"+appendix)
            modification_vector = tf.sign(tanh_arg, name="modification_vector"+appendix)
            # self.compute_boundary_state works as step function in forward pass
            # and as hard sigm in backward pass 
            boundary_state, old_emb_idx, slice_start = self.debug_compute_boundary_state(hard_sigm_arg,
                                                               idx,
                                                               appendix) 

            # Next operations implement c^l_t vector modification and h^l_t computing according to (2) and (3)
            # Since compute_boundary_state is the step function in forward pass
            # (if argument is greater than zero z^l_t = 1, otherwise z^l_t = 0)
            # equation (2) can be implemented either using tf.cond op
            # or via summing of all options multiplied flag which value is
            # equal to 0 or 1. I preferred the second variant because it doesn't involve
            # splitting input into batches and processing them separately.
            # In this algorithm I used 3 flags: update_flag, copy_flag and flush_flag
            # update_flag = 1 if UPDATE and 0 otherwise
            # copy_flag = 1 if COPY and 0 otherwise
            # flush_flag = 1 if FLUSH and 0 otherwise
            # flags, gates and vectors are transposed for broadcasting
            with tf.name_scope('boundary_operations'):
                update_flag = tf.transpose(tf.to_float(tf.logical_and(tf.equal(state[2],
                                                                               [[0.]],
                                                                               name="equal_state2_and_0_in_update_flag"+appendix),
                                                                      tf.equal(boundary_state_down,
                                                                               [[1.]],
                                                                               name="equal_boundary_state_down_and_1_in_update_flag"+appendix),
                                                                      name="logical_and_in_update_flag"+appendix),
                                                       name="to_float_in_update_flag"+appendix),
                                           name="update_flag"+appendix)
                copy_flag = tf.transpose(tf.to_float(tf.logical_and(tf.equal(state[2],
                                                                             [[0.]],
                                                                             name="equal_state2_and_0_in_copy_flag"+appendix),
                                                                    tf.equal(boundary_state_down,
                                                                             [[0.]],
                                                                             name="equal_boundary_state_down_and_0_in_copy_flag"+appendix),
                                                                    name="logical_and_in_copy_flag"+appendix),
                                                     name="to_float_in_copy_flag"+appendix),
                                         name="copy_flag"+appendix)
                flush_flag = tf.transpose(tf.to_float(tf.equal(state[2],
                                                               [[1.]],
                                                               name="equal_state2_and_1_in_flush_flag"+appendix),
                                                      name="to_float_in_flush_flag"+appendix),
                                          name="flush_flag"+appendix)
                # constant 'one' is used for building negations
                one = tf.constant([[1.]], name="one_constant"+appendix)
                tr_memory = tf.transpose(state[1], name="tr_memory"+appendix)
                tr_forget_gate = tf.transpose(forget_gate, name="tr_forget_gate"+appendix)
                tr_input_gate = tf.transpose(input_gate, name="tr_input_gate"+appendix)
                tr_output_gate = tf.transpose(output_gate, name="tr_output_gate"+appendix)
                tr_modification_vector = tf.transpose(modification_vector, name="tr_modification_vector"+appendix)
                # new memory computation
                update_term = tf.multiply(update_flag,
                                          tf.add(tf.multiply(tr_forget_gate,
                                                             tr_memory,
                                                             name="multiply_forget_and_memory_in_update_term"+appendix),
                                                 tf.multiply(tr_input_gate,
                                                             tr_modification_vector,
                                                             name="multiply_input_and_modification_in_update_term"+appendix),
                                                 name="add_in_update_term"+appendix),
                                          name="update_term"+appendix)
                copy_term = tf.multiply(copy_flag, tr_memory, name="copy_term"+appendix)
 
                
                flush_term = tf.multiply(flush_flag,
                                         tf.multiply(tr_input_gate,
                                                     tr_modification_vector,
                                                     name="multiply_input_and_modification_in_flush_term"+appendix),
                                         name="flush_term"+appendix)
                
                tr_new_memory = tf.add(tf.add(update_term,
                                              copy_term,
                                              name="add_update_and_copy_in_tr_new_memory"+appendix),
                                       flush_term,
                                       name="tr_new_memory"+appendix)
                new_memory = tf.transpose(tr_new_memory, name="new_memory"+appendix)
                # new hidden states computation
                tr_hidden = tf.transpose(state[0], name="tr_hidden"+appendix)
                copy_term = tf.multiply(copy_flag, tr_hidden, name="copy_term_for_hidden"+appendix)
                else_term = tf.multiply(tf.multiply(tf.subtract(one,
                                                                copy_flag,
                                                                name="subtract_in_else_term"+appendix),
                                                    tr_output_gate,
                                                    name="multiply_subtract_and_tr_output_gate_in_else_term"+appendix),
                                        tf.sign(tr_new_memory, name="tanh_in_else_term"+appendix),
                                        name="else_term"+appendix)
                new_hidden = tf.transpose(tf.add(copy_term, else_term, name="new_hidden"+appendix),
                                          name="new_hidden"+appendix)
                
                helper = {"L2_norm_of_hard_sigm_arg": L2_norm_of_hard_sigm_arg,
                          "old_emb_idx": old_emb_idx,
                          "slice_start": slice_start}
        return new_hidden, new_memory, boundary_state, helper
    
    def last_layer(self,
                   state,                 # A tuple of tensors containing h^L_{t-1}, c^L_{t-1} (L - total number of layers)
                   bottom_up,             # A tensor h^{L-1}_t  
                   boundary_state_down,   # A tensor z^{L-1}_t
                   appendix):
        # method implements operations (2) - (7) (shortly (1)) performed on the last layer
        # and returns 2 tensors: hidden state, memory state (h^L_t, c^L_t accordingly)
        
        with tf.name_scope('LSTM_layer_%s' % (self._num_layers-1)):
            # batch_size of processed data
            current_batch_size = bottom_up.get_shape().as_list()[0]
            # last layer idx
            last = self._num_layers-1


            # note: in several next operations tf.transpose method is applied repeatedly.
            #       It was used for broadcasting activation along vectors in same batches

            # this one cumputes a product z^{l-1}_t x h^{l-1}_t for formula (7)
            bottom_up_prepaired = tf.transpose(tf.multiply(tf.transpose(boundary_state_down,
                                                                        name="transposed_boundary_state_down_in_bottom_down_prepaired"+appendix),
                                                           tf.transpose(bottom_up,
                                                                        name="transposed_bottom_up_in_bottom_up_prepaired"+appendix),
                                                           name="multiply_in_bottom_up_prepaired"+appendix),
                                               name="bottom_up_prepaired"+appendix)

            # Matrix multiplications in formulas (5) - (7) and sum in argument of function f_slice
            # in formula (4) are united in one operation
            # Matrices U^l_l and W^l_{l-1} are concatenated into one matrix self.Matrices[last] 
            # and vectors h^l_{t-1} and  z^{l-1}_t x h^{l-1}_t are concatenated into vector X
            X = tf.concat([bottom_up_prepaired, state[0]],
                          1,
                          name="X"+appendix)                                          
            concat = tf.add(tf.matmul(X,
                                      self.Matrices[last].read_value(),
                                      name="matmul_in_concat"+appendix),
                            self.Biases[last].read_value(),
                            name="concat"+appendix)

            # following operations implement function vector implementation in formula (4)
            # and compute f^l_t, i^l_t, o^l_t, g^l_t and z^l_t
            # Note that that 'hard sigm' is omitted
            [sigmoid_arg, tanh_arg] = tf.split(concat, 
                                               [3*self._num_nodes[last], self._num_nodes[last]],
                                               axis=1,
                                               name="split_to_function_arguments"+appendix)                                          
            gate_concat = self.step_function(sigmoid_arg, "_gate_concat"+appendix)
            [forget_gate, input_gate, output_gate] = tf.split(gate_concat,
                                                              3,
                                                              axis=1,
                                                              name="split_to_gates_op")
            modification_vector = tf.sign(tanh_arg, name="modification_vector"+appendix)

            # Next operations implement c^l_t vector modification and h^l_t computing according to (2) and (3)
            # Check up detailed description in previous method's comments 
            # I used 2 flags: update_flag and copy_flag 
            # update_flag = 1 if UPDATE and 0 otherwise
            # copy_flag = 1 if COPY and 0 otherwise
            # flags, gates and vectors are transposed for broadcasting
            with tf.name_scope('boundary_operations'):
                update_flag = tf.transpose(tf.to_float(tf.equal(boundary_state_down,
                                                                1.,
                                                                name="equal_boundary_state_down_and_1_in_update_flag"+appendix),
                                                       name="to_float_in_update_flag"+appendix),
                                           name="update_flag"+appendix)
                # constant 'one' is used for building negations
                one = tf.constant([[1.]], name="one_constant"+appendix)
                copy_flag = tf.subtract(one, update_flag, name="copy_flag"+appendix)
                tr_memory = tf.transpose(state[1], name="tr_memory"+appendix)
                tr_forget_gate = tf.transpose(forget_gate, name="tr_forget_gate"+appendix)
                tr_input_gate = tf.transpose(input_gate, name="tr_input_gate"+appendix)
                tr_output_gate = tf.transpose(output_gate, name="tr_output_gate"+appendix)
                tr_modification_vector = tf.transpose(modification_vector, name="tr_modification_gate"+appendix)
                # new memory computation
                update_term = tf.multiply(update_flag,
                                          tf.add(tf.multiply(tr_forget_gate,
                                                             tr_memory,
                                                             name="multiply_forget_and_memory_in_update_term"+appendix),
                                                 tf.multiply(tr_input_gate,
                                                             tr_modification_vector,
                                                             name="multiply_input_and_modification_in_update_term"+appendix),
                                                 name="add_in_update_term"+appendix),
                                          name="update_term"+appendix)
                copy_term = tf.multiply(copy_flag, tr_memory, name="copy_term"+appendix)
                tr_new_memory = tf.add(update_term,
                                       copy_term,
                                       name="tr_new_memory"+appendix)
                new_memory = tf.transpose(tr_new_memory, name="new_memory"+appendix)
                # new hidden states computation
                tr_hidden = tf.transpose(state[0], name="tr_hidden"+appendix)
                copy_term = tf.multiply(copy_flag, tr_hidden, name="copy_term_for_hidden"+appendix)
                else_term = tf.multiply(tf.multiply(tf.subtract(one,
                                                                copy_flag,
                                                                name="subtract_in_else_term"+appendix),
                                                    tr_output_gate,
                                                    name="multiply_subtract_and_tr_output_gate_in_else_term"+appendix),
                                        tf.sign(tr_new_memory, name="tanh_in_else_term"+appendix),
                                        name="else_term"+appendix)
                new_hidden = tf.transpose(tf.add(copy_term, else_term, name="new_hidden"+appendix),
                                          name="new_hidden"+appendix)
        return new_hidden, new_memory
     
    
    def compute_boundary_state(self,
                               X,
                               appendix):
        # Elementwise calculates step function 
        # During backward pass works as hard sigm
        with self._graph.gradient_override_map({"Sign": "HardSigmoid"}):
            X = tf.sign(X, name="sign_func_in_compute_boundary"+appendix)       
        X = tf.divide(tf.add(X,
                             tf.constant([[1.]]),
                             name="add_in_compute_boundary_state"+appendix),
                      2.,
                      name="output_of_compute_boundary_state"+appendix)
        return X
    
    def debug_compute_boundary_state(self,
                                     X,
                                     layer_idx,
                                     appendix):
        # Elementwise calculates step function 
        # During backward pass works as hard sigm
        shape = X.get_shape().as_list()
        idx = tf.mod(self._global_step, 30, name="mod_debug_compute_boundary_state"+appendix)
        slice_start = tf.concat([[idx], tf.constant([layer_idx])],
                                        0,
                                        name="slice_start"+appendix)
        reshaped_slice = tf.reshape(tf.slice(self.debug_boundaries,
                                                     slice_start,
                                                     [1, 1],
                                                     name="slice_from_debug_boundaries"+appendix),
                                            [1, 1],
                                            name="reshaped_slice"+appendix)
        return_value = tf.tile(reshaped_slice,
                                       shape,
                                       name="fixed_boundaries"+appendix)
        return return_value, [idx], slice_start
    
    def iteration(self, inp, state, iter_idx, appendix):
        # This function implements processing of one character embedding by HM_LSTM
        # 'inp' is one character embedding
        # 'state' is network state from previous layer
        # Method returns: new state of the network which includes hidden states,
        # memory states and boundary states for all layers; concatenated boundaries for all
        # layers ([batch_size, self._num_layers-1])
        
        with tf.name_scope('iteration_%s'%iter_idx):

            num_layers = self._num_layers
            new_state = list()
            boundaries = list()

            # batch_size of processed data
            current_batch_size = state[0][0].get_shape().as_list()[0]
            # activated_boundary_states variable is used as boundary_state_down
            # argument on the first layer
            activated_boundary_states = tf.constant(1.,
                                                    shape=[current_batch_size, 1],
                                                    name="activated_boundary_states_in_iteration_function"+appendix)

            new_appendix_templ = appendix + "_layernum"

            # The first layer is calculated outside the loop
            hidden, memory, boundary, helper = self.not_last_layer(0,
                                                                   iter_idx,
                                                                   state[0],
                                                                   inp,
                                                                   state[1][0],
                                                                   activated_boundary_states,
                                                                   new_appendix_templ+str(0))

            not_last_layer_helpers = list()
            not_last_layer_helpers.append(helper)
            new_state.append((hidden, memory, boundary))
            boundaries.append(boundary)
            # All layers except for the first and the last ones
            if num_layers > 2:
                for idx in range(num_layers-2):
                    hidden, memory, boundary, helper = self.not_last_layer(idx+1,
                                                                           iter_idx,
                                                                          state[idx+1],
                                                                          hidden,
                                                                          state[idx+2][0],
                                                                          boundary,
                                                                          new_appendix_templ+str(idx+1))
                    not_last_layer_helpers.append(helper)
                    new_state.append((hidden, memory, boundary))
                    boundaries.append(boundary)
            hidden, memory = self.last_layer(state[-1],
                                             hidden,
                                             boundary,
                                             new_appendix_templ+str(self._num_layers-1))
            new_state.append((hidden, memory))
            L2_norm_of_hard_sigm_arg_list = list()
            helper = {"L2_norm_of_hard_sigm_arg": tf.concat([helper["L2_norm_of_hard_sigm_arg"] for helper in not_last_layer_helpers],
                                                            1,
                                                            name="L2_norm_of_hard_sigm_arg_for_all_layers"+appendix),
                      "old_emb_idx": tf.concat([helper['old_emb_idx'] for helper in not_last_layer_helpers],
                                               0,
                                               name='old_emb_idx'+appendix),
                      "slice_start": tf.stack([helper['slice_start'] for helper in not_last_layer_helpers],
                                              name="slice_start"+appendix)}
            return new_state, tf.concat(boundaries, 1, name="iteration_boundaries_output"+appendix), helper
    
    def embedding_module(self,
                         inputs,
                         appendix):
        # This function embeds input one-hot encodded character vector into a vector of dimension embedding_size
        # For computation acceleration inputs are concatenated before being multiplied on self.embedding_weights
        with tf.name_scope('embedding_module'):
            current_num_unrollings = len(inputs)
            inputs = tf.concat(inputs,
                               0,
                               name="inputs_concat_in_embedding_module"+appendix)
            embeddings = tf.matmul(inputs,
                                   self.embedding_weights,
                                   name="concatenated_embeddings_in_embedding_module"+appendix)
            split = tf.split(embeddings,
                            current_num_unrollings,
                            axis=0,
                            name="embedding_module_output"+appendix)
            return split
    
    
    def RNN_module(self,
                   embedded_inputs,
                   saved_state,
                   appendix):
        # This function implements processing of embedded inputs by HM_LSTM
        # Function returns: state of recurrent neural network after last character processing 'state',
        # hidden states obtained on each character 'saved_hidden_states' and boundaries on each layer 
        # on all characters.
        # Method returns 'state' state of network after last iteration (list of tuples (one tuple for each
        # layer), tuple contains hidden state ([batch_size, self._num_nodes[idx]]), memory state
        # ([batch_size, self._num_nodes[idx]]) and boundary state ([batch_size, 1])), list of concatenated along batch 
        # dimension hidden states for all layers 'saved_hidden_states' (each element of the list is tensor of dim 
        # [batch_size*num_unrollings, self._num_nodes[idx]]); a tensor containing L2 norms of hidden states
        # of shape [batch_size, num_unrollings, num_layers]
        with tf.name_scope('RNN_module'):
            # 'saved_hidden_states' is a list of self._num_layers elements. idx-th element of the list is 
            # a concatenation of hidden states on idx-th layer along chunk of input text.
            saved_hidden_states = list()
            for _ in range(self._num_layers):
                saved_hidden_states.append(list())

            # 'saved_iteration_boundaries' is a list
            saved_iteration_boundaries = list()
            new_appendix_templ = appendix + '_unrolling'
            state = saved_state
            iteration_helpers = list()
            for emb_idx, emb in enumerate(embedded_inputs):
                state, iteration_boundaries, helper = self.iteration(emb, state, emb_idx, new_appendix_templ+str(emb_idx))
                iteration_helpers.append(helper)
                saved_iteration_boundaries.append(iteration_boundaries)
                for layer_state, saved_hidden in zip(state, saved_hidden_states):
                    saved_hidden.append(layer_state[0])
                    
            # computing l2 norm of hidden states
            with tf.name_scope('L2_norm'):
                # all hidden states are packed to form a tensor of shape 
                # [batch_size, num_unrollings]
                L2_norm_by_layers = list()
                current_batch_size = saved_hidden_states[0][0].get_shape().as_list()[0]
                shape = [current_batch_size, len(embedded_inputs)]

                for layer_idx, saved_hidden in enumerate(saved_hidden_states):
                    stacked_for_L2_norm = tf.stack(saved_hidden,
                                                   axis=1,
                                                   name="stacked_hidden_states_on_layer%s"%layer_idx+appendix)
                    L2_norm_by_layers.append(tf.reshape(self.L2_norm(stacked_for_L2_norm,
                                                                     2,
                                                                     "_for_layer%s" % layer_idx+appendix),
                                                        shape,
                                                        name="L2_norm_for_layer%s" % layer_idx+appendix))
                L2_norm = tf.stack(L2_norm_by_layers, axis=2, name="L2_norm"+appendix)

            for idx, layer_saved in enumerate(saved_hidden_states):
                saved_hidden_states[idx] = tf.concat(layer_saved, 0, name=("hidden_concat_in_RNN_module_on_layer%s"%idx)+appendix)

            helper = {"L2_norm_of_hard_sigm_arg": tf.stack([helper["L2_norm_of_hard_sigm_arg"] for helper in iteration_helpers],
                                                           axis=1,
                                                           name="L2_norm_of_hard_sigm_arg_for_all_iterations"+appendix),
                      "all_boundaries": tf.stack(saved_iteration_boundaries,
                                                 axis=1,
                                                 name="stack_of_boundaries"+appendix),
                      "L2_norm_of_hidden_states": L2_norm,
                      "old_emb_idx": tf.stack([helper['old_emb_idx'] for helper in iteration_helpers],
                                              name="old_emb_idx"+appendix),
                      "slice_start": tf.stack([helper['slice_start'] for helper in iteration_helpers],
                                              name="slice_start"+appendix)}
            return state, saved_hidden_states, helper
            
    
    def output_module(self,
                      hidden_states,
                      appendix):
        with tf.name_scope('output_module'):
            concat = tf.concat(hidden_states, 1, name="total_concat_hidden"+appendix)
            output_module_gates = tf.transpose(self.step_function(tf.matmul(concat,
                                                                            self.output_module_gates_weights,
                                                                            name="matmul_in_output_module_gates"+appendix),
                                                                  "_sigmoid_in_output_module_gates"+appendix),
                                               name="output_module_gates"+appendix)
            output_module_gates = tf.split(output_module_gates,
                                           self._num_layers,
                                           axis=0,
                                           name="split_of_output_module_gates"+appendix)
            tr_gated_hidden_states = list()
            for idx, hidden_state in enumerate(hidden_states):
                tr_hidden_state = tf.transpose(hidden_state, name=("tr_hidden_state_total_%s"%idx)+appendix)
                tr_gated_hidden_states.append(tf.multiply(output_module_gates[idx],
                                                          tr_hidden_state,
                                                          name=("tr_gated_hidden_states_%s"%idx)+appendix))
            gated_hidden_states = tf.transpose(tf.concat(tr_gated_hidden_states,
                                                         0,
                                                         name="concat_in_gated_hidden_states"+appendix),
                                               name="gated_hidden_states"+appendix)
            return tf.add(tf.matmul(gated_hidden_states,
                                    self.output_weights,
                                    name="matmul_in_logits_output"+appendix),
                          self.output_bias,
                          name="logits"+appendix)
        
        
        
    def construct_boundary_test_line(self):
        test_line = list()
        samples = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
        def add1(index):
            test_line.append(list(samples[index]))
        def add2(index):
            test_line.append(list(samples[index]))
            test_line.append(list(samples[index]))
        add2(0)
        add1(2)
        add2(0)
        add1(1)
        add2(0)
        add1(3)
        add2(0)
        add2(2)
        add2(0)
        add2(1)
        add2(0)
        add2(3)
        add2(0)
        add2(3)
        add1(3)
        add2(0)
        add2(0)
        return test_line
    
    def change_gate_chunk(self,
                      layer_idx,
                      gate_idx,
                          coeff):
        positive_coeffs = tf.constant(1., shape=[self._num_nodes[layer_idx]], name="positive_coeffs_operation_for_layer%s_gate%s"%(layer_idx, gate_idx))
        other_coeffs = tf.constant(coeff, shape=[self._num_nodes[layer_idx]], name="other_coeffs_operation_for_layer%s_gate%s"%(layer_idx, gate_idx))
        list_for_coeffs = list()
        for _ in range(4):
            list_for_coeffs.append(positive_coeffs)
        list_for_coeffs[gate_idx] = other_coeffs
        if layer_idx != self._num_layers-1:
            list_for_coeffs.append(tf.constant(1., shape=[1], name="coeff_for_hard_sigm_operation_for_layer%s_gate%s"%(layer_idx, gate_idx)))
        coeffs = tf.concat(list_for_coeffs, 0, name="coeffs_for_layer%s_gate%s"%(layer_idx, gate_idx))
        modification_operation = tf.assign(self.Matrices[layer_idx],
                                           tf.multiply(self.Matrices[layer_idx],
                                                       coeffs,
                                                       name="multiply_in_modification_operation_for_layer%s_gate%s"%(layer_idx, gate_idx)),
                                           name="modification_operation_for_layer%s_gate%s"%(layer_idx, gate_idx))
        return modification_operation
    
    def change_flow_chunk(self,
                   layer_idx,
                   direction,
                    coeff):
        list_of_coeffs = list()
        
        if layer_idx != self._num_layers - 1:
            number_of_directions_on_layer = 3
        else:
            number_of_directions_on_layer = 2
        for i in range(-1, number_of_directions_on_layer-1):
            if i != direction:
                current_coeff = 1.
            else:
                current_coeff = coeff
            if not ((layer_idx == 0) and (i == -1)):
                list_of_coeffs.append(tf.constant(current_coeff,
                                                  shape=[self._num_nodes[layer_idx+i]],
                                                  name="intermediate_coeffs_for_layer%s_direction%s"%(layer_idx, i)))
            else:
                print(i)
                list_of_coeffs.append(tf.constant(current_coeff,
                                                  shape=[self._embedding_size],
                                                  name="intermediate_coeffs_for_layer%s_direction%s"%(layer_idx, i)))                
        coeffs = tf.concat(list_of_coeffs,
                               0,
                               name="coeffs_for_layer%s_direction%s"%(layer_idx, direction))
        modification_operation = tf.assign(self.Matrices[layer_idx],
                                           tf.transpose(tf.multiply(tf.transpose(self.Matrices[layer_idx],
                                                                                 name="transpose_in_modification_operation_for_layer%s_direction%s"%(layer_idx, direction)),
                                                                    coeffs,
                                                                    name="multiply_in_modification_operation_for_layer%s_direction%s"%(layer_idx, direction)),
                                                        name="back_transpose_in_modification_operation_for_layer%s_direction%s"%(layer_idx, direction)),
                                           name="modification_operation_for_layer%s_direction%s"%(layer_idx, direction))
        return modification_operation        
                
    def modifify_matrices(self,
                          gates_to_modify,
                          directions_to_modify):
        mod_ops = list()
        if gates_to_modify is not None:
            for gate in gates_to_modify:
                mod_ops.append(self.change_gate_chunk(*gate))
        if directions_to_modify is not None:
            for direction in directions_to_modify:
                mod_ops.append(self.change_flow_chunk(*direction))
        if len(mod_ops) > 0:
            return [tf.group(*mod_ops, name="final_operation")]
        else:
            return []
    
    def do_nothing(self,
                   gates_to_modify,
                   directions_to_modify):
        mod_ops = list()
        if gates_to_modify is not None:
            for gate in gates_to_modify:
                mod_ops.append(self.Matrices[gate[0]])
        if directions_to_modify is not None:
            for direction_to_modify in directions_to_modify:
                mod_ops.append(self.Matrices[direction_to_modify[0]])
        return tf.group(*mod_ops, name="empty_final_operation")
    
    def __init__(self,
                 batch_size,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 num_unrollings,
                 num_layers,
                 num_nodes,
                 init_slope,
                 slope_growth,
                 slope_half_life,
                 train_text,
                 valid_text,
                 gates_to_modify=None,
                 directions_to_modify=None,
                 embedding_size=2):
        self._results = list()
        self._batch_size = batch_size
        self._vocabulary = vocabulary
        self._vocabulary_size = len(vocabulary)
        self._characters_positions_in_vocabulary = characters_positions_in_vocabulary
        self._num_unrollings = num_unrollings
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._init_slope = init_slope
        self._slope_half_life = slope_half_life
        self._slope_growth = slope_growth
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
                         "init_slope": 8,
                         "slope_growth": 9,
                         "slope_half_life": 10,
                         "embedding_size": 11,
                         "type": 12}
        self._graph = tf.Graph()
        
        self._last_num_steps = 0
        with self._graph.as_default(): 
            with tf.name_scope('train'):
                self._global_step = tf.Variable(0, name="global_step")
            with self._graph.device('/gpu:0'):
                debug_boundaries = self.construct_boundary_test_line()
                self.debug_boundaries = tf.constant(debug_boundaries, name="debug_boundaries")
                # embedding module variables
                self.embedding_weights = tf.Variable(tf.ones([self._vocabulary_size, self._embedding_size],
                                                             name="embeddings_matrix_initialize"),
                                                     trainable=True,
                                                     name="embeddings_matrix_variable")
                
                # RNN module variables
                self.Matrices = list()
                self.Biases = list()
                
                # tensor name templates for HM_LSTM parameters
                init_matr_name = "HM_LSTM_matrix_%s_initializer"
                init_bias_name = "HM_LSTM_bias_%s_initializer" 
                matr_name = "HM_LSTM_matrix_%s"
                bias_name = "HM_LSTM_bias_%s"
                
                self.Matrices.append(tf.Variable(tf.ones([self._embedding_size + self._num_nodes[0] + self._num_nodes[1],
                                                          4 * self._num_nodes[0] + 1],
                                                         name=init_matr_name%0),
                                                 trainable=False,
                                                 name=matr_name%0))
                self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[0] + 1],
                                                        name=init_bias_name%0),
                                               trainable=False,
                                               name=bias_name%0))
                if self._num_layers > 2:
                    for i in range(self._num_layers - 2):
                        self.Matrices.append(tf.Variable(tf.ones([self._num_nodes[i] + self._num_nodes[i+1] + self._num_nodes[i+2],
                                                                  4 * self._num_nodes[i+1] + 1],
                                                                 name=init_matr_name%(i+1)),
                                                         trainable=False,
                                                         name=matr_name%(i+1)))
                        self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[i+1] + 1],
                                                                name=init_bias_name%(i+1)),
                                                       trainable=False,
                                                       name=bias_name%(i+1)))
                self.Matrices.append(tf.Variable(tf.ones([self._num_nodes[-1] + self._num_nodes[-2],
                                                          4 * self._num_nodes[-1]],
                                                         name=init_matr_name%(self._num_layers-1)),
                                                 trainable=False,
                                                 name=matr_name%(self._num_layers-1)))     
                self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[-1]],
                                                        name=init_bias_name%(self._num_layers-1)),
                                               trainable=False,
                                               name=bias_name%(self._num_layers-1)))

                dim_classifier_input = sum(self._num_nodes)
                
                # output module variables
                # output module gates weights (w^l vectors in (formula (11)))
                self.output_module_gates_weights = tf.Variable(tf.ones([dim_classifier_input, self._num_layers],
                                                                       name="output_gates_weights_initializer"),
                                                               trainable=False,
                                                               name="output_gates_weights")
                # classifier 
                self.output_weights = tf.Variable(tf.ones([dim_classifier_input, self._vocabulary_size],
                                                          name="output_weights_initializer"),
                                                  trainable=False,
                                                  name="output_weights")
                self.output_bias = tf.Variable(tf.zeros([self._vocabulary_size], name="output_bias_initializer"),
                                               trainable=False,
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
                    for i in range(self._num_layers-1):
                        saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]],
                                                                 name=saved_state_init_templ%(i, 0)),
                                                        trainable=False,
                                                        name=saved_state_templ%(i, 0)),
                                            tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]],
                                                                 name=saved_state_init_templ%(i, 1)),
                                                        trainable=False,
                                                        name=saved_state_templ%(i, 1)),
                                            tf.Variable(tf.zeros([self._batch_size, 1],
                                                                 name=saved_state_init_templ%(i, 2)),
                                                        trainable=False,
                                                        name=saved_state_templ%(i, 2))))
                    saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]],
                                                             name=saved_state_init_templ%(self._num_layers-1, 0)),
                                                    trainable=False,
                                                    name=saved_state_templ%(self._num_layers-1, 0)),
                                        tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]],
                                                             name=saved_state_init_templ%(self._num_layers-1, 1)),
                                                    trainable=False,
                                                    name=saved_state_templ%(self._num_layers-1, 1))))


                    # slope annealing trick
                    slope = tf.add(tf.constant(self._init_slope, name="init_slope_const"),
                                   tf.to_float((self._global_step / tf.constant(self._slope_half_life,
                                                                                dtype=tf.int32,
                                                                                name="slope_half_life_const")),
                                               name="to_float_in_slope_init") * tf.constant(self._slope_growth, name="slope_growth"),
                                   name="slope")

                    @tf.RegisterGradient("HardSigmoid")
                    def hard_sigm_grad(op,                # op is operation for which gradient is computed
                                       grad):             # loss partial gradients with respect to op outputs
                        # This function is added for implememting straight-through estimator as described in
                        # 3.3 paragraph of fundamental paper. It is used during backward pass for replacing
                        # tf.sign function gradient. 'hard sigm' function derivative is 0 from minus
                        # infinity to -1/a, a/2 from -1/a to 1/a and 0 from 1/a to plus infinity. Since in
                        # compute_boundary_state function for implementing step function tf.sign product is
                        # divided by 2, in hard_sigm_grad output gradient is equal to 'a', not to 'a/2' from
                        # -1/a to 1/a in order to compensate mentioned multiplication in compute_boundary_state
                        # function
                        op_input = op.inputs[0]
                        # slope is parameter 'a' in 'hard sigm' function
                        mask = tf.to_float(tf.logical_and(tf.greater_equal(op_input, -1./ slope, name="greater_equal_in_hard_sigm_mask"),
                                                          tf.less(op_input, 1. / slope, name="less_in_hard_sigm_mask"),
                                                          name="logical_and_in_hard_sigm_mask"),
                                           name="mask_in_hard_sigm")
                        return tf.multiply(slope,
                                           tf.multiply(grad,
                                                       mask,
                                                       name="grad_mask_multiply_in_hard_sigm"),
                                           name="hard_sigm_grad_output")

                    # appendix is used for constructing of tensor name. It is appended to tensor name
                    # to indicate to which part of graph operation belongs
                    appendix = "_train"
                    with tf.name_scope('matrix_modification'):
                        mod_op = [tf.cond(tf.equal(self._global_step, 0, name="equal_in_mod_op_init"+appendix),
                                              true_fn=lambda: self.modifify_matrices(gates_to_modify, directions_to_modify),
                                              false_fn=lambda: self.do_nothing(gates_to_modify, directions_to_modify),
                                              name="cond_in_mod_op_init"+appendix)]

                    with tf.control_dependencies(mod_op):
                        embedded_inputs = self.embedding_module(train_inputs, appendix)
                        state, hidden_states, train_helper = self.RNN_module(embedded_inputs, saved_state, appendix)
                        logits = self.output_module(hidden_states, appendix)
                    
                        self.old_emb_idx = tf.reshape(train_helper['old_emb_idx'], [-1], name="old_emb_idx")
                        self.slice_start = train_helper['slice_start']
                    
                        self.L2_train = tf.reshape(tf.slice(train_helper["L2_norm_of_hidden_states"],
                                                        [0, 0, 0],
                                                        [1, 10, 1],
                                                        name="slice_for_L2"+appendix),
                                               [-1],
                                               name="L2"+appendix)

                        self.save_list = list()
                        save_list_templ = "save_list_assign_layer%s_number%s"
                        for i in range(self._num_layers-1):
                            self.save_list.append(tf.assign(saved_state[i][0],
                                                   state[i][0],
                                                   name=save_list_templ%(i, 0)))
                            self.save_list.append(tf.assign(saved_state[i][1],
                                                   state[i][1],
                                                   name=save_list_templ%(i, 1)))
                            self.save_list.append(tf.assign(saved_state[i][2],
                                                   state[i][2],
                                                   name=save_list_templ%(i, 2)))
                        self.save_list.append(tf.assign(saved_state[-1][0],
                                               state[-1][0],
                                               name=save_list_templ%(self._num_layers-1, 0)))
                        self.save_list.append(tf.assign(saved_state[-1][1],
                                               state[-1][1],
                                               name=save_list_templ%(self._num_layers-1, 1)))

                        """skip operation"""
                        self._skip_operation = tf.group(*self.save_list, name="skip_operation")

                        with tf.control_dependencies(self.save_list):
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
                        self._learning_rate = tf.train.exponential_decay(10.0,
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
                    for i in range(self._num_layers-1):
                        saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[i]],
                                                                        name=saved_state_init_templ%(i, 0)),
                                                               trainable=False,
                                                               name=saved_state_templ%(i, 0)),
                                                   tf.Variable(tf.zeros([1, self._num_nodes[i]],
                                                                        name=saved_state_init_templ%(i, 1)),
                                                               trainable=False,
                                                               name=saved_state_templ%(i, 1)),
                                                   tf.Variable(tf.zeros([1, 1],
                                                                        name=saved_state_init_templ%(i, 2)),
                                                               trainable=False,
                                                               name=saved_state_templ%(i, 2))))
                    saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[-1]],
                                                                    name=saved_state_init_templ%(self._num_layers-1, 0)),
                                                           trainable=False,
                                                           name=saved_state_templ%(self._num_layers-1, 0)),
                                               tf.Variable(tf.zeros([1, self._num_nodes[-1]],
                                                                    name=saved_state_init_templ%(self._num_layers-1, 1)),
                                                           trainable=False,
                                                           name=saved_state_templ%(self._num_layers-1, 1))))

                    # validation initializer. 
                    validation_initializer_list = list()
                    for saved_layer_sample_state in saved_sample_state:
                        for tensor in saved_layer_sample_state:
                            validation_initializer_list.append(tensor)
                    validation_initilizer = tf.variables_initializer(validation_initializer_list, name="validation_initializer")
                    """PLACEHOLDER sample input"""
                    self._sample_input = tf.placeholder(tf.float32,
                                                        shape=[1, self._vocabulary_size],
                                                        name="sample_input_placeholder")

                    reset_list_templ = "reset_list_assign_layer%s_number%s"
                    saved_state_init_templ = "saved_state_layer%s_number%s_initializer"
                    reset_list = list()
                    for i in range(self._num_layers-1):
                        reset_list.append(tf.assign(saved_sample_state[i][0],
                                                    tf.zeros([1, self._num_nodes[i]],
                                                             name=saved_state_init_templ%(i, 0)),
                                                    name=reset_list_templ%(i, 0)))
                        reset_list.append(tf.assign(saved_sample_state[i][1],
                                                    tf.zeros([1, self._num_nodes[i]],
                                                             name=saved_state_init_templ%(i, 1)),
                                                    name=reset_list_templ%(i, 1)))
                        reset_list.append(tf.assign(saved_sample_state[i][2],
                                                    tf.zeros([1, 1],
                                                             name=saved_state_init_templ%(i, 2)),
                                                    name=reset_list_templ%(i, 2)))
                    reset_list.append(tf.assign(saved_sample_state[-1][0],
                                                tf.zeros([1, self._num_nodes[-1]],
                                                         name=saved_state_init_templ%(self._num_layers-1, 0)),
                                                name=reset_list_templ%(self._num_layers-1, 0)))
                    reset_list.append(tf.assign(saved_sample_state[-1][1],
                                                tf.zeros([1, self._num_nodes[-1]],
                                                         name=saved_state_init_templ%(self._num_layers-1, 1)),
                                                name=reset_list_templ%(self._num_layers-1, 1)))
                    #reset sample state
                    self._reset_sample_state = tf.group(*reset_list, name="reset_sample_state")
 
                    #reset sample state
                    self._reset_sample_state = tf.group(*reset_list, name="reset_sample_state")

                    appendix = "_validation"
                    sample_embedded_inputs = self.embedding_module([self._sample_input], appendix)
                    sample_state, sample_hidden_states, validation_helper = self.RNN_module(sample_embedded_inputs,
                                                                                            saved_sample_state,
                                                                                            appendix)
                    sample_logits = self.output_module(sample_hidden_states, appendix) 

                    sample_save_list = list()
                    save_list_templ = "save_sample_list_assign_layer%s_number%s"
                    for i in range(self._num_layers-1):
                        sample_save_list.append(tf.assign(saved_sample_state[i][0],
                                                          sample_state[i][0],
                                                          name=save_list_templ%(i, 0)))
                        sample_save_list.append(tf.assign(saved_sample_state[i][1],
                                                          sample_state[i][1],
                                                          name=save_list_templ%(i, 1)))
                        sample_save_list.append(tf.assign(saved_sample_state[i][2],
                                                          sample_state[i][2],
                                                          name=save_list_templ%(i, 2)))
                    sample_save_list.append(tf.assign(saved_sample_state[-1][0],
                                                      sample_state[-1][0],
                                                      name=save_list_templ%(self._num_layers-1, 0)))
                    sample_save_list.append(tf.assign(saved_sample_state[-1][1],
                                                      sample_state[-1][1],
                                                      name=save_list_templ%(self._num_layers-1, 1)))

                    with tf.control_dependencies(sample_save_list):
                        """sample prediction"""
                        self._sample_prediction = tf.nn.softmax(sample_logits, name="sample_prediction") 
                        self.L2_hidden_states_validation = tf.reshape(validation_helper["L2_norm_of_hidden_states"], [-1], name="L2_hidden_validation")
                        self.boundary = tf.reshape(validation_helper["all_boundaries"], [-1], name="sample_boundary")
                        self.sigm_arg = tf.reshape(validation_helper["L2_norm_of_hard_sigm_arg"], [-1], name="sample_sigm_arg")

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


model = HM_LSTM(1,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 1,
                 3,
                 [3, 4, 5],
                 .001,               # init_slope
                 0.001,                  # slope_growth
                 100,
                 train_text,
                 valid_text,
                #gates_to_modify=[[0, 1, -1.]],
                directions_to_modify=[[0, -1, 0.]],
                embedding_size=6)



logdir = "HM_LSTM/logging/debug_summary_log"
summary_dict={'summary_collection_frequency': 1,
              'summary_tensors': 'self._global_step'}
model.run(1,                # number of times learning_rate is decreased
          0.9,              # a factor by which learning_rate is decreased
            100,            # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            50,             # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            3,              # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            1,              # when train point is obtained validation may be performed
            20,             # when train point percent is calculated results got on averaging_number chunks are averaged
          fixed_number_of_steps=35,
            print_intermediate_results=True,
            add_operations = ['self.save_list'],
          print_steps=[i for i in range(30)],
          block_validation=True,
          summarizing_logdir=logdir,
           debug=True)





