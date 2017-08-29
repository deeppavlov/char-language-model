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




class LSTM(MODEL):
    
    
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
            last_hidden_states = list()
            state = saved_state
            for emb_idx, emb in enumerate(embedded_inputs):
                state  = self.iteration(emb, state, emb_idx)
                last_hidden_states.append(state[-1][0])
            last_hidden_states = tf.concat(last_hidden_states, 0, name="hidden_concat_in_RNN_module")

            return state, last_hidden_states
            
    
    def output_module(self,
                      hidden_state):
        with tf.name_scope('output_module'):
            output_embeddings = tf.nn.relu(tf.add(tf.matmul(hidden_state,
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
        
    
    def __init__(self,
                 batch_size,
                 vocabulary,
                 characters_positions_in_vocabulary,
                 num_unrollings,
                 num_layers,
                 num_nodes,
                 train_text,
                 valid_text,
                 embedding_size=128,
                 output_embedding_size=1024,
                 init_parameter=1e-6,               # init_parameter is used for balancing stddev in matrices initialization
                                                  # and initial learning rate
                 matr_init_parameter=10000.):
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
                         "averaging_number":7,
                         "embedding_size": 8,
                         "output_embedding_size": 9,
                         "init_parameter": 10,
                         "matr_init_parameter": 11,
                         "type": 12}
        self._graph = tf.Graph()
        
        self._last_num_steps = 0
        with self._graph.as_default(): 
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
                init_matr_name = "LSTM_matrix_%s_initializer"
                init_bias_name = "LSTM_bias_%s_initializer" 
                matr_name = "LSTM_matrix_%s"
                bias_name = "LSTM_bias_%s"
                
                self.Matrices.append(tf.Variable(tf.truncated_normal([self._embedding_size + self._num_nodes[0],
                                                                      4 * self._num_nodes[0]],
                                                                     mean=0.,
                                                                     stddev=math.sqrt(self._init_parameter*matr_init_parameter/(self._embedding_size+self._num_nodes[0])),
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
                                                                             stddev=math.sqrt(self._init_parameter*matr_init_parameter/(self._num_nodes[i]+self._num_nodes[i+1])),
                                                                             name=init_matr_name%(i+1)),
                                                         name=matr_name%(i+1)))
                        self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[i+1]],
                                                                name=init_bias_name%(i+1)),
                                                       name=bias_name%(i+1)))

                dim_classifier_input = self._num_nodes[-1]
                
                # output module variables
                # classifier 
                self.output_embedding_weights = tf.Variable(tf.truncated_normal([dim_classifier_input, self._output_embedding_size],
                                                                                stddev=math.sqrt(self._init_parameter*matr_init_parameter/dim_classifier_input),
                                                                                name="output_embedding_weights_initializer"),
                                                            name="output_embedding_weights")
                self.output_embedding_bias = tf.Variable(tf.zeros([self._output_embedding_size], name="output_bias_initializer"),
                                                         name="output_bias")
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
                    dimensions.append(self._embedding_size + self._num_nodes[0])
                    if self._num_layers > 1:
                        for i in range(self._num_layers-1):
                            dimensions.append(self._num_nodes[i] + self._num_nodes[i+1])
                    dimensions.append(sum(self._num_nodes))
                    max_dimension = max(dimensions)
                    
                    self._learning_rate = tf.train.exponential_decay(160.*math.sqrt(self._init_parameter/max_dimension),
                                                                     self._global_step,
                                                                     self._half_life,
                                                                     self._decay,
                                                                     staircase=True,
                                                                     name="learning_rate")
                    regularizer = tf.contrib.layers.l2_regularizer(.5)
                    l2_loss = regularizer(self.output_embedding_weights)
                    output_embedding_weights_shape = self.output_embedding_weights.get_shape().as_list()
                    l2_divider = float(output_embedding_weights_shape[0] * output_embedding_weights_shape[1])
                    #optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)                    
                    optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
                    gradients, v = zip(*optimizer.compute_gradients(self._loss + l2_loss / l2_divider))
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
        metadata.append(self._embedding_size)
        metadata.append(self._output_embedding_size)
        metadata.append(self._init_parameter)
        metadata.append(self._matr_init_parameter)
        metadata.append('LSTM_stacked')
        return metadata


