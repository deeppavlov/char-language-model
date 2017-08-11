
# coding: utf-8

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
import sys
import os
if not os.path.isfile('model_module.py') or not os.path.isfile('plot_module.py'):
    current_path = os.path.dirname(os.path.abspath('__file__'))
    additional_path = '/'.join(current_path.split('/')[:-1])
    sys.path.append(additional_path)
from model_module import MODEL
from model_module import BatchGenerator
from model_module import characters

version = sys.version_info[0]


# This class implements hierarchical LSTM described in the paper https://arxiv.org/pdf/1609.01704.pdf
# All variables names and formula indices are taken from mentioned article
# notation A^i stands for A with upper index i
# notation A_i stands for A with lower index i
# notation A^i_j stands for A with upper index i and lower index j
class HM_LSTM(MODEL):
    
        
    def L2_norm(self,
                tensor,
                dim,
                appendix,
                keep_dims=True):
        with tf.name_scope('L2_norm'+appendix):
            square = tf.square(tensor, name="square_in_L2_norm")
            reduced = tf.reduce_mean(square,
                                     dim,
                                     keep_dims=keep_dims,
                                     name="reduce_mean_in_L2_norm")
            return tf.sqrt(reduced, name="L2_norm")
    
    
    def not_last_layer(self,
                       idx,                   # layer number (from 0 to self._num_layers - 1)
                       iter_idx,
                       state,                 # A tuple of tensors containing h^l_{t-1}, c^l_{t-1} and z^l_{t-1}
                       bottom_up,             # A tensor h^{l-1}_t  
                       top_down,              # A tensor h^{l+1}_{t-1}
                       boundary_state_down,   # A tensor z^{l-1}_t
                       new_boundary):

        # method implements operations (2) - (7) (shortly (1)) performed on idx-th layer
        # ONLY NOT FOR LAST LAYER! Last layer computations are implemented in self.last_layer method
        # and returns 3 tensors: hidden state, memory state
        # and boundary state (h^l_t, c^l_t and z^l_t accordingly)
        
        with tf.name_scope('LSTM_layer_%s'%(idx)):
            # batch_size of processed data
            current_batch_size = bottom_up.get_shape().as_list()[0]

            one = tf.constant([[1.]], name="one_constant")


            # following operation computes a product z^l_{t-1} x h^{l+1}_{t-1} for formula (6)
            top_down_prepaired = tf.multiply(state[2],
                                             top_down,
                                             name="top_down_prepaired")

            # this one cumputes a product z^{l-1}_t x h^{l-1}_t for formula (7)
            bottom_up_prepaired = tf.multiply(boundary_state_down,
                                              bottom_up,
                                              name="bottom_up_prepaired")
            
            boundary_state_reversed = tf.subtract(one, state[2], name="boundary_state_reversed")
            state0_prepaired = tf.multiply(boundary_state_reversed,
                                           state[0],
                                           name="state0_prepaired")
            

            # Matrix multiplications in formulas (5) - (7) and sum in argument of function f_slice
            # in formula (4) are united in one operation
            # Matrices U^l_l, U^l_{l+1} and W^l_{l-1} are concatenated into one matrix self.Matrices[idx]
            # and vectors h^l_{t-1}, z^l_{t-1} x h^{l+1}_{t-1} and  z^{l-1}_t x h^{l-1}_t are 
            # concatenated into vector X
            X = tf.concat([bottom_up_prepaired, state0_prepaired, top_down_prepaired],
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
            L2_forget_gate = self.L2_norm(forget_gate, None, 'forget_gate_layer%s' % idx, keep_dims=False)
            modification_vector = tf.tanh(tanh_arg, name="modification_vector")

            boundary_state = new_boundary

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
                update_flag = tf.to_float(tf.logical_and(tf.equal(state[2],
                                                                  [[0.]],
                                                                  name="equal_state2_and_0_in_update_flag"),
                                                         tf.equal(boundary_state_down,
                                                                  [[1.]],
                                                                  name="equal_boundary_state_down_and_1_in_update_flag"),
                                                         name="logical_and_in_update_flag"),
                                           name="update_flag")
                copy_flag = tf.to_float(tf.logical_and(tf.equal(state[2],
                                                                [[0.]],
                                                                name="equal_state2_and_0_in_copy_flag"),
                                                       tf.equal(boundary_state_down,
                                                                [[0.]],
                                                                name="equal_boundary_state_down_and_0_in_copy_flag"),
                                                       name="logical_and_in_copy_flag"),
                                        name="copy_flag")
                flush_flag = tf.to_float(tf.equal(state[2],
                                                  [[1.]],
                                                  name="equal_state2_and_1_in_flush_flag"),
                                         name="flush_flag")
                # constant 'one' is used for building negations
                one = tf.constant([[1.]], name="one_constant")
                memory = state[1]
                # new memory computation
                update_term = tf.multiply(update_flag,
                                          tf.add(tf.multiply(forget_gate,
                                                             memory,
                                                             name="multiply_forget_and_memory_in_update_term"),
                                                 tf.multiply(input_gate,
                                                             modification_vector,
                                                             name="multiply_input_and_modification_in_update_term"),
                                                 name="add_in_update_term"),
                                          name="update_term")
                copy_term = tf.multiply(copy_flag, memory, name="copy_term")

                
                flush_term = tf.multiply(flush_flag,
                                         tf.multiply(input_gate,
                                                     modification_vector,
                                                     name="multiply_input_and_modification_in_flush_term"),
                                         name="flush_term")
                
                new_memory = tf.add(tf.add(update_term,
                                           copy_term,
                                           name="add_update_and_copy_in_new_memory"),
                                    flush_term,
                                    name="new_memory")
                # new hidden states computation
                hidden = state[0]
                copy_term = tf.multiply(copy_flag, hidden, name="copy_term_for_hidden")
                else_term = tf.multiply(tf.multiply(tf.subtract(one,
                                                                copy_flag,
                                                                name="subtract_in_else_term"),
                                                    output_gate,
                                                    name="multiply_subtract_and_output_gate_in_else_term"),
                                        tf.tanh(new_memory, name="tanh_in_else_term"),
                                        name="else_term")
                new_hidden = tf.add(copy_term, else_term, name="new_hidden")
                
                helper = {"L2_forget_gate": L2_forget_gate}
        return new_hidden, new_memory, boundary_state, helper
    
    def last_layer(self,
                   state,                 # A tuple of tensors containing h^L_{t-1}, c^L_{t-1} (L - total number of layers)
                   bottom_up,             # A tensor h^{L-1}_t  
                   boundary_state_down):   # A tensor z^{L-1}_t
        # method implements operations (2) - (7) (shortly (1)) performed on the last layer
        # and returns 2 tensors: hidden state, memory state (h^L_t, c^L_t accordingly)
        
        with tf.name_scope('LSTM_layer_%s' % (self._num_layers-1)):
            # batch_size of processed data
            current_batch_size = bottom_up.get_shape().as_list()[0]
            # last layer idx
            last = self._num_layers-1

            # this one cumputes a product z^{l-1}_t x h^{l-1}_t for formula (7)
            bottom_up_prepaired = tf.multiply(boundary_state_down,
                                              bottom_up,
                                              name="bottom_up_prepaired")

            # Matrix multiplications in formulas (5) - (7) and sum in argument of function f_slice
            # in formula (4) are united in one operation
            # Matrices U^l_l and W^l_{l-1} are concatenated into one matrix self.Matrices[last] 
            # and vectors h^l_{t-1} and  z^{l-1}_t x h^{l-1}_t are concatenated into vector X
            X = tf.concat([bottom_up_prepaired, state[0]],
                          1,
                          name="X")                                          
            concat = tf.add(tf.matmul(X,
                                      self.Matrices[last],
                                      name="matmul_in_concat"),
                            self.Biases[last],
                            name="concat")

            # following operations implement function vector implementation in formula (4)
            # and compute f^l_t, i^l_t, o^l_t, g^l_t and z^l_t
            # Note that that 'hard sigm' is omitted
            [sigmoid_arg, tanh_arg] = tf.split(concat, 
                                               [3*self._num_nodes[last], self._num_nodes[last]],
                                               axis=1,
                                               name="split_to_function_arguments")                                          
            gate_concat = tf.sigmoid(sigmoid_arg, name="gate_concat")
            [forget_gate, input_gate, output_gate] = tf.split(gate_concat,
                                                              3,
                                                              axis=1,
                                                              name="split_to_gates_op")
            L2_forget_gate = self.L2_norm(forget_gate,
                                          None,
                                          "forget_gate_layer%s"%(self._num_layers-1),
                                          keep_dims=False)
            modification_vector = tf.tanh(tanh_arg, name="modification_vector")

            # Next operations implement c^l_t vector modification and h^l_t computing according to (2) and (3)
            # Check up detailed description in previous method's comments 
            # I used 2 flags: update_flag and copy_flag 
            # update_flag = 1 if UPDATE and 0 otherwise
            # copy_flag = 1 if COPY and 0 otherwise
            # flags, gates and vectors are transposed for broadcasting
            with tf.name_scope('boundary_operations'):
                update_flag = tf.to_float(tf.equal(boundary_state_down,
                                                   1.,
                                                   name="equal_boundary_state_down_and_1_in_update_flag"),
                                          name="update_flag")
                # constant 'one' is used for building negations
                one = tf.constant([[1.]], name="one_constant")
                copy_flag = tf.subtract(one, update_flag, name="copy_flag")
                memory = state[1]
                # new memory computation
                update_term = tf.multiply(update_flag,
                                          tf.add(tf.multiply(forget_gate,
                                                             memory,
                                                             name="multiply_forget_and_memory_in_update_term"),
                                                 tf.multiply(input_gate,
                                                             modification_vector,
                                                             name="multiply_input_and_modification_in_update_term"),
                                                 name="add_in_update_term"),
                                          name="update_term")
                copy_term = tf.multiply(copy_flag, memory, name="copy_term")
                new_memory = tf.add(update_term,
                                    copy_term,
                                    name="new_memory")
                # new hidden states computation
                hidden = state[0]
                copy_term = tf.multiply(copy_flag, hidden, name="copy_term_for_hidden")
                else_term = tf.multiply(tf.multiply(tf.subtract(one,
                                                                copy_flag,
                                                                name="subtract_in_else_term"),
                                                    output_gate,
                                                    name="multiply_subtract_and_output_gate_in_else_term"),
                                        tf.tanh(new_memory, name="tanh_in_else_term"),
                                        name="else_term")
                new_hidden = tf.add(copy_term, else_term, name="new_hidden")
                helper = {"L2_forget_gate": L2_forget_gate}
        return new_hidden, new_memory, helper
     
    
    def iteration(self, inp, state, iter_idx, new_boundaries):
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
            helpers = list()

            # batch_size of processed data
            current_batch_size = state[0][0].get_shape().as_list()[0]
            # activated_boundary_states variable is used as boundary_state_down
            # argument on the first layer
            activated_boundary_states = tf.constant(1.,
                                                    shape=[current_batch_size, 1],
                                                    name="activated_boundary_states_in_iteration_function")

            # The first layer is calculated outside the loop

            hidden = inp
            boundary = activated_boundary_states
            # All layers except for the first and the last ones
            for idx in range(num_layers-1):
                hidden, memory, boundary, helper = self.not_last_layer(idx,
                                                                       iter_idx,
                                                                       state[idx],
                                                                       hidden,
                                                                       state[idx+1][0],
                                                                       boundary,
                                                                       new_boundaries[idx])
                helpers.append(helper)
                new_state.append((hidden, memory, boundary))
                boundaries.append(boundary)
            hidden, memory, helper = self.last_layer(state[-1],
                                                     hidden,
                                                     boundary)
            helpers.append(helper)
            new_state.append((hidden, memory))
            helper = {"L2_forget_gate": tf.stack([helper["L2_forget_gate"] for helper in helpers],
                                                 name="L2_forget_gate_for_iteration%s"%iter_idx)}
            return new_state, tf.concat(boundaries, 1, name="iteration_boundaries_output"), helper
        
    def compute_boundary_states(self,
                                inputs):
        with tf.name_scope('compute_boundary_states'):
            current_num_unrollings = len(inputs)
            inputs = tf.concat(inputs,
                               0,
                               name="inputs_concat_in_embedding_module")
            with tf.name_scope('first_level'):
                after_multiplication = tf.multiply(inputs,
                                                   self.first_level_delimeters,
                                                   name='multiply_in_compute_boundary_states')
                boundary_states_1 = tf.reduce_sum(after_multiplication,
                                                  axis=1,
                                                  keep_dims=True,
                                                  name='boundary_states_concat')
                boundary_states_1 = tf.split(boundary_states_1,
                                             current_num_unrollings,
                                             axis=0,
                                             name="boundary_states")
            with tf.name_scope('second_level'):
                after_multiplication = tf.multiply(inputs,
                                                   self.second_level_delimeters,
                                                   name='multiply_in_compute_boundary_states')
                boundary_states_2 = tf.reduce_sum(after_multiplication,
                                                  axis=1,
                                                  keep_dims=True,
                                                  name='boundary_states_concat')
                boundary_states_2 = tf.split(boundary_states_2,
                                             current_num_unrollings,
                                             axis=0,
                                             name="boundary_states")
            return [boundary_states_1, boundary_states_2]             
    
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
    
    def process_saved_boundary(self, boundary, name_of_scope):
        with tf.name_scope(name_of_scope):
            plus_one = tf.add(boundary, 1., name='plus_one')
            processed_boundary = tf.subtract(plus_one, 1., name='processed_boundary')
            return processed_boundary
   
    def retrieve_state(self, saved_state):
        processed_boundaries = list()
        with tf.name_scope('retrieving_state'):
            for layer_idx in range(self._num_layers - 1):
                processed_boundaries.append(self.process_saved_boundary(saved_state[layer_idx][2], 'boundary_%s_processing'%layer_idx))
        processed_state = list()
        for layer_idx in range(self._num_layers - 1):
            processed_state.append((saved_state[layer_idx][0],
                                    saved_state[layer_idx][1],
                                    processed_boundaries[layer_idx]))
        processed_state.append(saved_state[self._num_layers-1])
        return processed_state, processed_boundaries
    
    def RNN_module(self,
                   embedded_inputs,
                   saved_state,
                   boundaries):
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
            state = saved_state
            iteration_helpers = list()
            for emb_idx, emb in enumerate(embedded_inputs):
                input_boundaries = [boundaries[0][emb_idx], boundaries[1][emb_idx]]
                state, iteration_boundaries, helper = self.iteration(emb, state, emb_idx, input_boundaries)
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
                                                   name="stacked_hidden_states_on_layer%s"%layer_idx)
                    L2_norm_by_layers.append(tf.reshape(self.L2_norm(stacked_for_L2_norm,
                                                                     2,
                                                                     "_for_layer%s" % layer_idx),
                                                        shape,
                                                        name="L2_norm_for_layer%s" % layer_idx))
                L2_norm = tf.stack(L2_norm_by_layers, axis=2, name="L2_norm")

            for idx, layer_saved in enumerate(saved_hidden_states):
                saved_hidden_states[idx] = tf.concat(layer_saved, 0, name="hidden_concat_in_RNN_module_on_layer%s"%idx)

            helper = {"all_boundaries": tf.stack(saved_iteration_boundaries,
                                                 axis=1,
                                                 name="stack_of_boundaries"),
                      "L2_norm_of_hidden_states": L2_norm,
                      "L2_forget_gate": tf.stack([helper['L2_forget_gate'] for helper in iteration_helpers],
                                                 name="L2_forget_gate_all")}
            return state, saved_hidden_states, helper
            
    
    def output_module(self,
                      hidden_states):
        # hidden_states is list of hidden_states by layer, concatenated along batch dimension
        with tf.name_scope('output_module'):
            concat = tf.concat(hidden_states, 1, name="total_concat_hidden")
            output_module_gates = tf.sigmoid(tf.matmul(concat,
                                                       self.output_module_gates_weights,
                                                       name="matmul_in_output_module_gates"),
                                             name="output_module_gates")
            output_module_gates = tf.split(output_module_gates,
                                           self._num_layers,
                                           axis=1,
                                           name="split_of_output_module_gates")
            gated_hidden_states = list()
            for idx, hidden_state in enumerate(hidden_states):
                gated_hidden_states.append(tf.multiply(output_module_gates[idx],
                                                       hidden_state,
                                                       name="gated_hidden_states_%s"%idx))
            gated_hidden_states = tf.concat(gated_hidden_states,
                                            1,
                                            name="gated_hidden_states")
            output_embeddings = tf.nn.relu(tf.add(tf.matmul(gated_hidden_states,
                                                            self.output_embedding_weights,
                                                            name="matmul_in_output_embeddings"),
                                                  self.output_embedding_bias,
                                                  name="xW_plus_b_in_output_embeddings"),
                                           name="output_embeddings")
            return tf.add(tf.matmul(output_embeddings,
                                    self.output_weights,
                                    name="matmul_in_logits"),
                          self.output_bias,
                          name="logits")
        
    def compute_perplexity(self, probabilities):
        with tf.name_scope('perplexity'):
            ln2 = tf.log(2., name="ln2")
            too_small_mask = tf.to_float(tf.less(probabilities,
                                                 1e-10,
                                                 name="less_too_small_mask"),
                                         name="too_small_mask")
            not_small_mask = tf.subtract(1., too_small_mask, name="not_small_mask")
            too_small_term = tf.multiply(too_small_mask, 1e-10, name="too_small_term")
            not_small_term = tf.multiply(not_small_mask, probabilities, name="not_small_term")
            probabilities = tf.add(too_small_term, not_small_term, name="probabilities")
            log_probabilities = tf.divide(tf.log(probabilities, name="log_in_compute_probability"), ln2, name="log_probabilities")
            neg_probabilities = tf.negative(probabilities, name="negative_in_compute_probability")
            multiply = tf.multiply(neg_probabilities, log_probabilities, name="multiply_in_compute_probability")
            entropy = tf.reduce_sum(multiply, axis=1, name="entropy")
            perplexity = tf.exp(tf.multiply(ln2, entropy, name="multiply_in_perplexity"), name="perplexity")
            return tf.reduce_mean(perplexity, name="mean_perplexity")
        
    def create_delimeters(self):
        first_level_delimeters = [' ', '\n', '\t']
        second_level_delimeters = list()
        length = len(self._vocabulary)
        for char in self._vocabulary:
            relative_lowercase = ord(char) - ord('a')
            relative_uppercase = ord(char) - ord('A')
            it_is_not_lowercase = (relative_lowercase < 0) or (relative_lowercase > 25)
            it_is_not_uppercase = (relative_uppercase < 0) or (relative_uppercase > 25)
            it_is_not_apostrophe = (char != "'")
            if it_is_not_lowercase and it_is_not_uppercase and it_is_not_apostrophe:
                it_is_not_first_level_delimeter = True
                for delimeter in first_level_delimeters:
                    it_is_not_first_level_delimeter = it_is_not_first_level_delimeter and (delimeter != char)
                if it_is_not_first_level_delimeter:
                    second_level_delimeters.append(char) 
        first_level_delimeters.extend(second_level_delimeters)
        base = tf.constant([0.]*length, name='base')
        first_level = base
        second_level = base
        with tf.name_scope('delimeters'):
            one_hot_list = [0.]*length
            for idx, delimeter in enumerate(first_level_delimeters):
                one_hot_list[self._vocabulary.index(delimeter)] = 1.
            first_level = tf.constant(one_hot_list, name='first_level_delimeters')
            one_hot_list = [0.]*length
            for idx, delimeter in enumerate(second_level_delimeters):
                one_hot_list[self._vocabulary.index(delimeter)] = 1.
            second_level = tf.constant(one_hot_list, name='second_level_delimeters')
        return first_level, second_level        

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
                 embedding_size=128,
                 output_embedding_size=1024,
                 init_parameter=1.,               # init_parameter is used for balancing stddev in matrices initialization
                                                  # and initial learning rate
                 matr_init_parameter=1000.):     
                                                   
                                                   
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
        self._output_embedding_size = output_embedding_size
        self._init_parameter = init_parameter
        self._matr_init_parameter = matr_init_parameter

        self._indices = {"batch_size": 0,
                         "num_unrollings": 1,
                         "num_layers": 2,
                         "num_nodes": 3,
                         "half_life": 4,
                         "decay": 5,
                         "num_steps": 6,
                         "averaging_number": 7,
                         "embedding_size": 8,
                         "output_embedding_size": 9,
                         "init_parameter": 10,
                         "matr_init_parameter": 11,
                         "type": 12}
        self._graph = tf.Graph()
        
        self._last_num_steps = 0
        with self._graph.as_default(): 
            tf.set_random_seed(1)
            self.first_level_delimeters, self.second_level_delimeters = self.create_delimeters()
            with tf.name_scope('train'):
                self._global_step = tf.Variable(0, trainable=False, name="global_step")
            with self._graph.device('/gpu:0'):
                # embedding module variables
                self.embedding_weights = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._embedding_size],
                                                                         stddev = math.sqrt(self._init_parameter*matr_init_parameter/self._vocabulary_size),
                                                                         name="embeddings_matrix_initialize"), 
                                                     name="embeddings_matrix_variable")
                
                # RNN module variables
                self.Matrices = list()
                self.Biases = list()
                
                # tensor name templates for HM_LSTM parameters
                init_matr_name = "HM_LSTM_matrix_%s_initializer"
                init_bias_name = "HM_LSTM_bias_%s_initializer"
                bias_name = "HM_LSTM_bias_%s" 
                matr_name = "HM_LSTM_matrix_%s"
                
                def compute_dim_and_bias(layer_idx):
                    output_dim = 4 * self._num_nodes[layer_idx]
                    if layer_idx == self._num_layers - 1:
                        input_dim = self._num_nodes[-1] + self._num_nodes[-2]
                    elif layer_idx == 0:
                        input_dim = self._embedding_size + self._num_nodes[0] + self._num_nodes[1]
                    else:
                        input_dim = self._num_nodes[layer_idx - 1] + self._num_nodes[layer_idx] + self._num_nodes[layer_idx+1]
                    stddev = math.sqrt(self._init_parameter*matr_init_parameter/input_dim)
                    return input_dim, output_dim, stddev
                
                for layer_idx in range(self._num_layers):
                    input_dim, output_dim, stddev = compute_dim_and_bias(layer_idx)
                    self.Biases.append(tf.Variable(tf.zeros([output_dim], name=init_bias_name%layer_idx),
                                                   name=bias_name%layer_idx))         
                    self.Matrices.append(tf.Variable(tf.truncated_normal([input_dim,
                                                                          output_dim],
                                                                         mean=0.,
                                                                         stddev=stddev,
                                                                         name=init_matr_name%0),
                                                     name=matr_name%layer_idx))     

                dim_classifier_input = sum(self._num_nodes)
                
                # output module variables
                # output module gates weights (w^l vectors in (formula (11)))
                self.output_module_gates_weights = tf.Variable(tf.truncated_normal([dim_classifier_input, self._num_layers],
                                                                                   stddev = math.sqrt(self._init_parameter*matr_init_parameter/dim_classifier_input),
                                                                                   name="output_gates_weights_initializer"),
                                                               name="output_gates_weights")
                # classifier 
                self.output_embedding_weights = tf.Variable(tf.truncated_normal([dim_classifier_input, self._output_embedding_size],
                                                                                stddev=math.sqrt(self._init_parameter*matr_init_parameter/dim_classifier_input),
                                                                                name="output_embedding_weights_initializer"),
                                                            name="output_embedding_weights")
                self.output_embedding_bias = tf.Variable(tf.zeros([self._output_embedding_size], name="output_bias_initializer"),
                                                         name="output_embedding_bias")
                self.output_weights = tf.Variable(tf.truncated_normal([self._output_embedding_size, self._vocabulary_size],
                                                                      stddev = math.sqrt(self._init_parameter*matr_init_parameter/self._output_embedding_size),
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
                    train_inputs_for_slice = tf.stack(train_inputs,
                                                      axis=1,
                                                      name="train_inputs_for_slice")
                    self.train_input_print = tf.reshape(tf.split(train_inputs_for_slice,
                                                                 [2, 1, self._batch_size-3],
                                                                 name="split_in_train_print")[1],
                                                        [self._num_unrollings, -1],
                                                        name="train_print")


                    self.saved_state = list()
                    # templates for saved_state tensor names
                    saved_state_init_templ = "saved_state_layer%s_number%s_initializer"
                    saved_state_templ = "saved_state_layer%s_number%s"
                    for i in range(self._num_layers-1):
                        self.saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]],
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
                    self.saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]],
                                                                  name=saved_state_init_templ%(self._num_layers-1, 0)),
                                                         trainable=False,
                                                         name=saved_state_templ%(self._num_layers-1, 0)),
                                             tf.Variable(tf.zeros([self._batch_size, self._num_nodes[-1]],
                                                                  name=saved_state_init_templ%(self._num_layers-1, 1)),
                                                         trainable=False,
                                                         name=saved_state_templ%(self._num_layers-1, 1))))


                    embedded_inputs = self.embedding_module(train_inputs)
                    retrieved_state, control_operations = self.retrieve_state(self.saved_state)
                    with tf.control_dependencies(control_operations):
                        self.boundaries = self.compute_boundary_states(train_inputs)
                    state, hidden_states, train_helper = self.RNN_module(embedded_inputs,
                                                                         retrieved_state,
                                                                         self.boundaries)
                    logits = self.output_module(hidden_states)
                    
                    self.L2_train = tf.reshape(tf.slice(train_helper["L2_norm_of_hidden_states"],
                                                        [0, 0, 0],
                                                        [1, 10, 1],
                                                        name="slice_for_L2"),
                                               [-1],
                                               name="L2")
                    
                    L2_forget_gate_reduced = tf.reduce_mean(train_helper['L2_forget_gate'],
                                                            axis=0,
                                                            name="L2_forget_gate_reduced")
                    
                    self.L2_forget_gate = tf.unstack(L2_forget_gate_reduced, name="L2_forget_gate_unstacked")
                    self.all_boundaries = train_helper['all_boundaries']
                    one_chunk_boundaries = tf.split(self.all_boundaries,
                                                    [2, 1, self._batch_size-3],
                                                    name='one_chunk_boundaries')[1]
                    self.boundaries_for_plot = tf.reshape(one_chunk_boundaries,
                                                          [self._num_unrollings, 2],
                                                          name='boundaries_for_plot')
                    flush_fractions_stacked = tf.reduce_mean(train_helper['all_boundaries'],
                                                             axis=[0, 1],
                                                             name='flush_fractions_stacked')
                    
                    self.flush_fractions = tf.split(flush_fractions_stacked,
                                                    self._num_layers-1,
                                                    axis=0,
                                                    name='flush_fractions')
                    
                    save_list = list()
                    save_list_templ = "save_list_assign_layer%s_number%s"
                    for i in range(self._num_layers-1):
                        save_list.append(tf.assign(self.saved_state[i][0],
                                                   state[i][0],
                                                   name=save_list_templ%(i, 0)))
                        save_list.append(tf.assign(self.saved_state[i][1],
                                                   state[i][1],
                                                   name=save_list_templ%(i, 1)))
                        save_list.append(tf.assign(self.saved_state[i][2],
                                                   state[i][2],
                                                   name=save_list_templ%(i, 2)))
                    save_list.append(tf.assign(self.saved_state[-1][0],
                                               state[-1][0],
                                               name=save_list_templ%(self._num_layers-1, 0)))
                    save_list.append(tf.assign(self.saved_state[-1][1],
                                               state[-1][1],
                                               name=save_list_templ%(self._num_layers-1, 1)))

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
                    
                    self._learning_rate = tf.train.exponential_decay(160.*math.sqrt(self._init_parameter/max_dimension),
                                                                     self._global_step,
                                                                     self._half_life,
                                                                     self._decay,
                                                                     staircase=True,
                                                                     name="learning_rate")
                    #optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
                    gradients, v = zip(*optimizer.compute_gradients(self._loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
                    """optimizer"""
                    self._optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=self._global_step)
                    """train prediction"""
                    self._train_prediction = tf.nn.softmax(logits, name="train_prediction")
                    print_list = tf.split(self._train_prediction, self._num_unrollings, name="print_list")
                    print_for_slice = tf.stack(print_list, axis=1, name="print_for_slice")
                    self.train_output_print = tf.reshape(tf.split(print_for_slice,
                                                                  [1, self._batch_size-1],
                                                                  name="split_in_train_print")[0],
                                                         [self._num_unrollings, -1],
                                                         name="train_print")
                    self.train_perplexity = self.compute_perplexity(self._train_prediction)
                    

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


                    sample_embedded_inputs = self.embedding_module([self._sample_input])
                    retrieved_state, control_operations = self.retrieve_state(saved_sample_state)
                    with tf.control_dependencies(control_operations):
                        self.sample_boundary_states = self.compute_boundary_states([self._sample_input])
                    sample_state, sample_hidden_states, validation_helper = self.RNN_module(sample_embedded_inputs,
                                                                                            retrieved_state,
                                                                                            self.sample_boundary_states)
                    sample_logits = self.output_module(sample_hidden_states) 

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
                        self.validation_perplexity = self.compute_perplexity(self._sample_prediction)
                # creating control dictionary
                all_vars = tf.global_variables()
                self.control_dictionary = dict()
                #print('building graph')
                for variable in all_vars:
                    
                    list_to_form_name = variable.name.split('/')
                    if ':' in list_to_form_name[-1]:
                        list_to_form_name[-1] = list_to_form_name[-1].split(':')[0]
                    if len(list_to_form_name) < 2:
                        name = list_to_form_name[0] 
                    else:
                        name = list_to_form_name[0] + '_' + list_to_form_name[-1]
                    norm = self.L2_norm(tf.to_float(variable,
                                                    name="to_float_in_control_dictionary_for_"+list_to_form_name[-1]),
                                        None,
                                        list_to_form_name[-1],
                                        keep_dims=False)
                    with tf.device('/cpu:0'):
                        self.control_dictionary[name] = tf.summary.scalar(name+'_sum', 
                                                                          norm)
                    #print(name, ': ', norm.get_shape().as_list())
            forget_template = 'self.L2_forget_gate[%s]'
            flush_fractions_template = 'self.flush_fractions[%s]'
            for layer_idx in range(self._num_layers):
                self.control_dictionary[forget_template % layer_idx] = tf.summary.scalar(forget_template % layer_idx +'_sum', 
                                                                                         self.L2_forget_gate[layer_idx])
            for layer_idx in range(self._num_layers-1):
                self.control_dictionary[flush_fractions_template % layer_idx] = tf.summary.scalar(flush_fractions_template % layer_idx +'_sum', 
                                                                                                  tf.reshape(self.flush_fractions[layer_idx],
                                                                                                             [],
                                                                                                             name='reshaping_flush_fractions_%s'%layer_idx))
            self.control_dictionary['loss'] = tf.summary.scalar('loss_sum', self._loss) 
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
        metadata.append(self._embedding_size)
        metadata.append(self._output_embedding_size)
        metadata.append(self._init_parameter)
        metadata.append(self._matr_init_parameter)
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
            if self._valid_size // num_strings < length:
                num_strings = self._valid_size // length
            for i in range(num_strings):
                start_positions.append(i* (self._valid_size // num_strings) + self._valid_size // num_strings // 2)
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
                    
            else:        
                _ = self._sample_prediction.eval({self._sample_input: b[0]})
        return text_list, boundaries_list
