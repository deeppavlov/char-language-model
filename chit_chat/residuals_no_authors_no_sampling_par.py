from __future__ import print_function
import numpy as np
import tensorflow as tf
from some_useful_functions import (construct, create_vocabulary,
                                   get_positions_in_vocabulary, char2vec, pred2vec, vec2char,
                                   char2id, id2char, flatten, get_available_gpus, device_name_scope,
                                   average_gradients, get_num_gpus_and_bs_on_gpus)


url = 'http://mattmahoney.net/dc/'


class LstmBatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, characters_positions_in_vocabulary):
        return np.reshape(char2vec(char, characters_positions_in_vocabulary), (1, 1, -1))

    @staticmethod
    def pred2vec(pred):
        return np.reshape(pred2vec(pred), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.characters_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id('\n', self.characters_positions_in_vocabulary)] = 1.0
        return batch

    def _zero_batch(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]], self.characters_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def char2batch(self, char):
        return np.stack(char2vec(char, self.characters_positions_in_vocabulary)), np.stack(self._zero_batch())

    def pred2batch(self, pred):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        return np.stack([batch]), np.stack([self._zero_batch()])

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)


def characters(probabilities, vocabulary):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c, vocabulary) for c in np.argmax(probabilities, 1)]


def batches2string(batches, vocabulary):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [u""] * batches[0].shape[0]
    for b in batches:
        s = [u"".join(x) for x in zip(s, characters(b, vocabulary))]
    return s


class Model(object):

    @classmethod
    def get_name(cls):
        return cls._name


class Lstm(Model):
    _name = 'lstm'

    @classmethod
    def check_kwargs(cls,
                     **kwargs):
        pass

    @classmethod
    def get_name(cls):
        return cls._name

    @staticmethod
    def get_special_args():
        return dict()

    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    def _lstm_layer(self, inp, conn_vec, state, layer_idx):
        with tf.name_scope('lstm_layer_%s' % layer_idx):
            matr = self._lstm_matrices[layer_idx]
            bias = self._lstm_biases[layer_idx]
            nn = self._num_nodes[layer_idx]
            # print('inp.shape:', inp.get_shape().as_list())
            # print('conn_vec.shape:', conn_vec.get_shape().as_list())
            # print('state[0].shape:', state[0].get_shape().as_list())
            x = tf.concat([tf.nn.dropout(inp, self.dropout_keep_prob), conn_vec, state[0]], 1, name='X')
            linear_res = tf.add(tf.matmul(x, matr, name='matmul'), bias, name='linear_res')
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3*nn, nn], axis=1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, (new_hidden_state, new_cell_state)

    def _rnn_iter(self, embedding, conn_vectors, all_states):
        with tf.name_scope('rnn_iter'):
            new_all_states = list()
            all_hidden_states = list()
            output = embedding
            for layer_idx, state in enumerate(all_states):
                output, state = self._lstm_layer(output, conn_vectors[layer_idx], state, layer_idx)
                new_all_states.append(state)
                all_hidden_states.append(state[0])
            return all_hidden_states, new_all_states

    def _embed(self, inputs):
        with tf.name_scope('embeddings'):
            num_unrollings = len(inputs)
            inputs = tf.concat(inputs, 0, name='concatenated_inputs')
            embeddings = tf.matmul(inputs, self._embedding_matrix, name='embeddings_stacked')
            return tf.split(embeddings, num_unrollings, 0, name='embeddings')

    def _output_module(self, all_hidden_states):
        with tf.name_scope('output_module'):

            # print('all_hidden_states:', all_hidden_states)
            hidden_by_layer = zip(*all_hidden_states)
            concatenated_along_time_dim = list()
            for layer_idx, layer_hidden in enumerate(hidden_by_layer):
                concatenated_along_time_dim.append(tf.concat(layer_hidden, 0, name='hidden_along_time_%s' % layer_idx))
            full_concat = tf.concat(concatenated_along_time_dim, 1, name='full_concat')
            gates = tf.sigmoid(tf.matmul(full_concat,
                                         self._output_gates_matrix,
                                         name='matmul_in_gates'),
                               name='gates_concat')
            gates = tf.split(gates, self._num_layers, axis=1)
            gated = 0
            for gate, hidden in zip(gates, concatenated_along_time_dim):
                gated += gate * hidden
            hs = gated
            for layer_idx in range(self._num_output_layers):
                #print('hs.shape:', hs.get_shape().as_list())
                hs = tf.add(
                    tf.matmul(hs,
                              self._output_matrices[layer_idx],
                              name='matmul_in_%s_output_layer' % layer_idx),
                    self._output_biases[layer_idx],
                    name='res_of_%s_output_layer' % layer_idx)
                if layer_idx < self._num_output_layers - 1:
                    hs = tf.nn.relu(hs)
        return hs

    def _create_connection_vectors(self, saved_for_connections, idx):
        with tf.name_scope('attention_%s' % idx):
            # print('\n_create_connection_vectors')
            # print('saved_for_connections:', saved_for_connections)

            # for_computing = tf.concat(saved_for_connections[-self._connection_visibility:], 0)
            # print('for_computing:', for_computing)
            # print('for_computing.shape:', for_computing.get_shape().as_list())
            # compressed = tf.tanh(tf.matmul(for_computing, self._compress_matrix) + self._compress_bias)
            # print('compressed:', compressed)
            # compressed_by_layer = tf.split(compressed, self._num_layers, axis=1)
            # print('compressed_by_layer:', compressed_by_layer)
            # connection_vectors = list()
            # for layer_idx, layer_conn_vec in enumerate(compressed_by_layer):
            #     connection_vectors.append(tf.concat(tf.split(layer_conn_vec, self._connection_visibility),
            #                                         1,
            #                                         name='connection_for_layer_%s' % layer_idx))
            # print('connection_vectors:', connection_vectors)
            return tf.split(tf.concat(saved_for_connections[-self._connection_visibility:],
                                      1),
                            self._num_layers, 1,
                            name='connection_vectors')

    @staticmethod
    def _extract_op_name(full_name):
        scopes_stripped = full_name.split('/')[-1]
        return scopes_stripped.split(':')[0]

    def _compose_save_list(self,
                           *pairs):
        #print('start')
        with tf.name_scope('save_list'):
            save_list = list()
            for pair in pairs:
                #print('pair:', pair)
                variables = flatten(pair[0])
                #print(variables)
                new_values = flatten(pair[1])
                for variable, value in zip(variables, new_values):
                    name = self._extract_op_name(variable.name)
                    save_list.append(tf.assign(variable, value, name='assign_save_%s' % name))
            return save_list

    def _recurrent_assign_loop_with_dependencies(self, variables, new_values, control_operations):
        with tf.control_dependencies(control_operations):
            new_control_op = tf.assign(variables[0], new_values[0])
            if len(variables) > 1:
                s_list = self._recurrent_assign_loop_with_dependencies(variables[1:], new_values[1:], [new_control_op])
            else:
                s_list = list()
            s_list.append(new_control_op)
            return s_list

    def _compose_save_list_secure(self,
                                 *pairs):
        with tf.name_scope('secure_save_list'):
            save_list = list()
            for pair in pairs:
                variables = flatten(pair[0])
                new_values = flatten(pair[1])
                s_list = self._recurrent_assign_loop_with_dependencies(variables, new_values, [])
                save_list.extend(s_list)
            return save_list

    def _compose_reset_list(self, *args):
        with tf.name_scope('reset_list'):
            reset_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                reset_list.append(
                    tf.assign(variable, tf.zeros(shape=shape, dtype=variable.dtype),
                              name='assign_reset_%s' % name))
            return reset_list

    def _compose_randomize_list(self, *args):
        with tf.name_scope('randomize_list'):
            randomize_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                assign_tensor = tf.truncated_normal(shape, stddev=1.)
                randomize_list.append(tf.assign(variable, assign_tensor, name='assign_reset_%s' % name))
            return randomize_list

    def _compute_lstm_matrix_parameters(self, idx):
        if idx == 0:
            print(self._num_nodes)
            input_dim = self._dim_compressed * self._connection_visibility + self._num_nodes[0] + self._embedding_size
        else:
            input_dim = (self._dim_compressed * self._connection_visibility +
                         self._num_nodes[idx-1] + self._num_nodes[idx])
        output_dim = 4 * self._num_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1./input_dim)
        return input_dim, output_dim, stddev

    def _compute_output_matrix_parameters(self, idx):
        if idx == 0:
            #print('self._num_nodes:', self._num_nodes)
            input_dim = self._num_nodes[-1]
        else:
            input_dim = self._num_output_nodes[idx-1]
        if idx == self._num_output_layers - 1:
            output_dim = self._vocabulary_size
        else:
            output_dim = self._num_output_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1. / input_dim)
        return input_dim, output_dim, stddev

    def _l2_loss(self, matrices):
        with tf.name_scope('l2_loss'):
            regularizer = tf.contrib.layers.l2_regularizer(.5)
            loss = 0
            for matr in matrices:
                loss += regularizer(matr)
            return loss * self._regularization_rate

    def _train_graph(self):

        tower_grads = list()
        preds = list()
        losses = list()

        for gpu_batch_size, gpu_name, device_inputs, device_labels in zip(
                self._batch_sizes_on_gpus, self._gpu_names, self._inputs_by_device, self._labels_by_device):
            with tf.device(gpu_name):
                with tf.name_scope(device_name_scope(gpu_name)):
                    with tf.name_scope('train'):
                        saved_states = list()
                        for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
                            saved_states.append(
                                (tf.Variable(
                                     tf.zeros([gpu_batch_size, layer_num_nodes]),
                                     trainable=False,
                                     name='saved_state_%s_%s' % (layer_idx, 0)),
                                 tf.Variable(
                                     tf.zeros([gpu_batch_size, layer_num_nodes]),
                                     trainable=False,
                                     name='saved_state_%s_%s' % (layer_idx, 1)))
                            )
                        saved_for_connections = list()
                        for idx in range(self._connection_visibility):
                            saved_for_connections.append(tf.Variable(tf.zeros([gpu_batch_size,
                                                                               self._num_layers * self._dim_compressed]),
                                                                     trainable=False,
                                                                     name='saved_for_connection_%s' % idx))

                        inputs = tf.unstack(device_inputs)

                        states = saved_states
                        for_connections = list(saved_for_connections)
                        embeddings = self._embed(inputs)
                        all_hidden_states = list()
                        for iter_idx, emb in enumerate(embeddings):
                            with tf.name_scope('iteration_%s' % iter_idx):
                                if iter_idx % self._connection_interval == 0:
                                    connection_vectors = self._create_connection_vectors(for_connections,
                                                                                         iter_idx)
                                hidden_states, states = self._rnn_iter(emb, connection_vectors, states)
                                all_hidden_states.append(hidden_states)
                                if iter_idx % self._connection_interval == 0:
                                    to_compress = tf.concat(hidden_states, 1, name='for_connections')
                                    for_connections.append(
                                        tf.tanh(tf.matmul(to_compress, self._compress_matrix) + self._compress_bias))

                        logits = self._output_module(all_hidden_states)

                        save_ops = self._compose_save_list([saved_states, states])
                        save_ops.extend(self._compose_save_list_secure(
                            [saved_for_connections, for_connections[-self._connection_visibility:]]))
                        with tf.control_dependencies(save_ops):
                            all_matrices = [self._embedding_matrix]
                            all_matrices.extend(self._lstm_matrices)
                            all_matrices.extend(self._output_matrices)
                            all_matrices.append(self._output_gates_matrix)
                            l2_loss = self._l2_loss(all_matrices)
                            loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=device_labels, logits=logits))
                            losses.append(loss)
                            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                            grads_and_vars = optimizer.compute_gradients(loss + l2_loss)
                            tower_grads.append(grads_and_vars)

                            # splitting concatenated results for different characters
                            concat_pred = tf.nn.softmax(logits)
                            preds.append(tf.split(concat_pred, self._num_unrollings))
        with tf.device('/cpu:0'):
            with tf.name_scope(device_name_scope('/cpu:0') + '_gradients'):
                grads_and_vars = average_gradients(tower_grads)
                grads, v = zip(*grads_and_vars)
                grads, _ = tf.clip_by_global_norm(grads, 1.)
                self.train_op = optimizer.apply_gradients(zip(grads, v))
                self._hooks['train_op'] = self.train_op

                # composing predictions
                preds_by_char = list()
                # print('preds:', preds)
                for one_char_preds in zip(*preds):
                    # print('one_char_preds:', one_char_preds)
                    preds_by_char.append(tf.concat(one_char_preds, 0))
                # print('len(preds_by_char):', len(preds_by_char))
                self.predictions = tf.concat(preds_by_char, 0)
                self._hooks['predictions'] = self.predictions
                # print('self.predictions.get_shape().as_list():', self.predictions.get_shape().as_list())
                l = 0
                for loss, gpu_batch_size in zip(losses, self._batch_sizes_on_gpus):
                    l += float(gpu_batch_size) / float(self._batch_size) * loss
                self.loss = l
                self._hooks['loss'] = self.loss

    def _validation_graph(self):
        with tf.name_scope('validation'):
            saved_sample_counter = tf.Variable(0, trainable=False, name='valid_counter')
            self.sample_input = tf.placeholder(tf.float32, shape=[1, 1, self._vocabulary_size])
            self._hooks['validation_inputs'] = self.sample_input
            sample_input = tf.reshape(self.sample_input, [1, -1])
            saved_sample_state = list()
            for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
                saved_sample_state.append(
                    (tf.Variable(
                        tf.zeros([1, layer_num_nodes]),
                        trainable=False,
                        name='saved_sample_state_%s_%s' % (layer_idx, 0)),
                     tf.Variable(
                         tf.zeros([1, layer_num_nodes]),
                         trainable=False,
                         name='saved_sample_state_%s_%s' % (layer_idx, 1)))
                )
            saved_sample_for_connections = tf.Variable(tf.zeros([self._connection_visibility, 1,
                                                                 self._num_layers * self._dim_compressed]),
                                                       trainable=False,
                                                       name='saved_sample_state')
            sample_for_connections = tf.unstack(saved_sample_for_connections)
            # print('\nsample_for_connections:', sample_for_connections)

            saved_sample_from_connections = list()
            for layer_idx in range(self._num_layers):
                saved_sample_from_connections.append(
                    tf.Variable(tf.zeros([1, self._dim_compressed * self._connection_visibility]),
                                trainable=False,
                                name='saved_sample_from_connections_%s' % layer_idx))

            reset_list = self._compose_reset_list(saved_sample_state,
                                                  saved_sample_for_connections,
                                                  saved_sample_from_connections)
            self.reset_sample_state = tf.group(*reset_list)
            self._hooks['reset_validation_state'] = self.reset_sample_state

            randomize_list = self._compose_randomize_list(saved_sample_state,
                                                          saved_sample_for_connections,
                                                          saved_sample_from_connections)
            self.randomize = tf.group(*randomize_list)
            self._hooks['randomize_sample_state'] = self.randomize

            it_is_time_to_reset_from_connections = tf.equal(tf.mod(saved_sample_counter, self._connection_interval), 0)
            sample_from_connections = tf.cond(
                it_is_time_to_reset_from_connections,
                true_fn=lambda: self._create_connection_vectors(sample_for_connections,
                                                                0),
                false_fn=lambda: saved_sample_from_connections)

            embeddings = self._embed([sample_input])
            sample_hidden_states, sample_state = self._rnn_iter(embeddings[0],
                                                                sample_from_connections,
                                                                saved_sample_state)
            sample_logits = self._output_module([sample_hidden_states])
            to_compress = tf.concat(sample_hidden_states, 1, name='validation_to_compress')
            new_for_connections = tf.tanh(tf.matmul(to_compress, self._compress_matrix) + self._compress_bias)

            sample_for_connections.append(new_for_connections)
            sample_for_connections = tf.stack(sample_for_connections)
            sample_for_connections = tf.cond(
                it_is_time_to_reset_from_connections,
                true_fn=lambda: tf.split(sample_for_connections, [1, self._connection_visibility], axis=0)[1],
                false_fn=lambda: tf.split(sample_for_connections, [self._connection_visibility, 1], axis=0)[0]
            )

            sample_counter = saved_sample_counter + 1
            sample_save_ops = self._compose_save_list([saved_sample_state, sample_state],
                                                      [saved_sample_for_connections, sample_for_connections],
                                                      [saved_sample_counter, sample_counter],
                                                      [saved_sample_from_connections, sample_from_connections])

            with tf.control_dependencies(sample_save_ops):
                self.sample_prediction = tf.nn.softmax(sample_logits)
                self._hooks['validation_predictions'] = self.sample_prediction

    def __init__(self,
                 batch_size=64,
                 num_layers=2,
                 num_nodes=None,
                 num_output_layers=1,
                 num_output_nodes=None,
                 dim_compressed=None,
                 vocabulary_size=None,
                 embedding_size=128,
                 connection_interval=3,
                 connection_visibility=3,
                 subsequence_length_in_intervals=7,
                 init_parameter=3.,
                 regularization_rate=.000003,
                 num_gpus=1,
                 regime='train'):

        if num_nodes is None:
            num_nodes = [100, 100]
        if num_output_nodes is None:
            num_output_nodes = []

        self._hooks = dict(inputs=None,
                           labels=None,
                           train_op=None,
                           learning_rate=None,
                           loss=None,
                           predictions=None,
                           validation_inputs=None,
                           validation_predictions=None,
                           reset_validation_state=None,
                           randomize_sample_state=None,
                           dropout=None,
                           saver=None)
        self._num_gpus = num_gpus
        self._batch_size = batch_size

        self._gpu_names = get_available_gpus()
        self._num_available_gpus = len(self._gpu_names)
        self._num_gpus, self._batch_sizes_on_gpus = get_num_gpus_and_bs_on_gpus(
            self._batch_size, self._num_gpus, self._num_available_gpus)
        self._num_gpus = num_gpus
        self._batch_sizes_on_gpus = self._batch_sizes_on_gpus

        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._num_output_layers = num_output_layers
        self._num_output_nodes = num_output_nodes
        self._connection_interval = connection_interval
        self._connection_visibility = connection_visibility
        self._subsequence_length_in_intervals = subsequence_length_in_intervals

        if dim_compressed is None:
            self._dim_compressed = sum(self._num_nodes) // (self._connection_visibility * self._num_layers)
        else:
            self._dim_compressed = dim_compressed

        self._num_unrollings = self._subsequence_length_in_intervals * self._connection_interval
        self._init_parameter = init_parameter
        self._regularization_rate = regularization_rate
        with tf.device('/cpu:0'):
            self.inputs = tf.placeholder(tf.float32,
                                         shape=[self._num_unrollings, self._batch_size, self._vocabulary_size])
            self.labels = tf.placeholder(tf.float32,
                                         shape=[self._num_unrollings * self._batch_size, self._vocabulary_size])
            self._hooks['inputs'] = self.inputs
            self._hooks['labels'] = self.labels

            inputs = tf.split(self.inputs, self._batch_sizes_on_gpus, 1)
            self._inputs_by_device = list()
            for device_inputs in inputs:
                self._inputs_by_device.append(tf.unstack(device_inputs))

            labels = tf.reshape(self.labels, shape=(self._num_unrollings, self._batch_size, self._vocabulary_size))
            labels = tf.split(labels, self._batch_sizes_on_gpus, 1)
            self._labels_by_device = list()
            for device_labels in labels:
                self._labels_by_device.append(tf.reshape(device_labels, [-1, self._vocabulary_size]))

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self.learning_rate

            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            self._hooks['dropout'] = self.dropout_keep_prob
            self._embedding_matrix = tf.Variable(
                tf.truncated_normal([self._vocabulary_size, self._embedding_size],
                                    stddev=self._init_parameter*np.sqrt(1./self._vocabulary_size)),
                name='embedding_matrix')

            self._lstm_matrices = list()
            self._lstm_biases = list()
            for layer_idx in range(self._num_layers):
                input_dim, output_dim, stddev = self._compute_lstm_matrix_parameters(layer_idx)
                self._lstm_matrices.append(tf.Variable(tf.truncated_normal([input_dim,
                                                                            output_dim],
                                                                           stddev=stddev),
                                                       name='lstm_matrix_%s' % layer_idx))
                self._lstm_biases.append(tf.Variable(tf.zeros([output_dim]), name='lstm_bias_%s' % layer_idx))

            input_dim = sum(self._num_nodes)
            output_dim = self._num_layers
            stddev = self._init_parameter * np.sqrt(1. / input_dim)
            self._output_gates_matrix = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                                        stddev=stddev),
                                                    name='output_gates_matrix')

            self._output_matrices = list()
            self._output_biases = list()
            for layer_idx in range(self._num_output_layers):
                input_dim, output_dim, stddev = self._compute_output_matrix_parameters(layer_idx)
                #print('input_dim:', input_dim)
                #print('output_dim:', output_dim)
                self._output_matrices.append(tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                                             stddev=stddev),
                                                         name='output_matrix_%s' % layer_idx))
                self._output_biases.append(tf.Variable(tf.zeros([output_dim])))
            input_dim = sum(self._num_nodes)
            stddev = self._init_parameter * np.sqrt(1. / input_dim)
            self._compress_matrix = tf.Variable(
                tf.truncated_normal([sum(self._num_nodes), self._dim_compressed * self._num_layers],
                                    stddev=stddev),
                name='compress_matrix')
            self._compress_bias = tf.Variable(tf.zeros([self._dim_compressed * self._num_layers]))
            saved_vars = dict()
            saved_vars['embedding_matrix'] = self._embedding_matrix
            for layer_idx, (lstm_matrix, lstm_bias) in enumerate(zip(self._lstm_matrices, self._lstm_biases)):
                saved_vars['lstm_matrix_%s' % layer_idx] = lstm_matrix
                saved_vars['lstm_bias_%s' % layer_idx] = lstm_bias
            saved_vars['output_gates_matrix'] = self._output_gates_matrix
            for layer_idx, (output_matrix, output_bias) in enumerate(zip(self._output_matrices, self._output_biases)):
                saved_vars['output_matrix_%s' % layer_idx] = output_matrix
                saved_vars['output_bias_%s' % layer_idx] = output_bias
            saved_vars['compress_matrix'] = self._compress_matrix
            saved_vars['compress_bias'] = self._compress_bias
            self.saver = tf.train.Saver(saved_vars, max_to_keep=None)
            self._hooks['saver'] = self.saver

        if regime == 'train':
            self._train_graph()
            self._validation_graph()
        if regime == 'inference':
            self._validation_graph()

    def get_default_hooks(self):
        return dict(self._hooks.items())

    def get_building_parameters(self):
        pass
