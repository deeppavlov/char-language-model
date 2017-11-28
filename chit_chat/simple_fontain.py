import numpy as np
import math
import re
import tensorflow as tf
from model import Model
from some_useful_functions import create_vocabulary, char2id, get_positions_in_vocabulary, construct, flatten


def char2vec(character_positions_in_vocabulary,
             char,
             speaker_flag_size=2,
             speaker_idx=0,
             bot_answer_flag=0,
             eod=False):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size + speaker_flag_size + 2), dtype=np.float32)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    vec[0, voc_size + speaker_idx] = 1.0
    vec[0, voc_size + speaker_flag_size] = float(bot_answer_flag)
    vec[0, voc_size + speaker_flag_size + 1] = float(eod)
    return vec


def char_2_base_vec(character_positions_in_vocabulary,
                    char):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float32)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    return vec


def add_flags_2_simple_text(input_file_name,
                            output_file_name,
                            eod_interval):
    input_f = open(input_file_name, 'r')
    lines = input_f.readlines()
    output_f = open(output_file_name, 'w')
    replicas_counter = 0
    for line in lines:
        output_f.write('<%s>' % (replicas_counter % 2))
        output_f.write(line)
        replicas_counter += 1
        if replicas_counter % eod_interval == 0:
            output_f.write('<EOD>')
    input_f.close()
    output_f.close()


def process_input_text(text):
    #print('text:', text)
    splitted = text.split('<')
    number_of_speakers = 0
    counter = 0
    new_text = ''
    eod_flags = list()
    speaker_flags = list()
    bot_answer_flags = list()
    for chunk in splitted:
        match = re.match("[0-9]+>", chunk)
        if match is not None:
            current_speaker_flag = int(match.group()[:-1])
            if current_speaker_flag >= number_of_speakers:
                number_of_speakers = current_speaker_flag + 1
            if current_speaker_flag == 0:
                current_bot_answer_flag = 1
            else:
                current_bot_answer_flag = 0
            span = match.span()
            string = chunk[span[1]:]
            new_text += string
            length = len(string)
            eod_flags.extend([0]*length)
            speaker_flags.extend([current_speaker_flag]*length)
            bot_answer_flags.extend([current_bot_answer_flag]*length)
            counter += length
        elif re.match("EOD>", chunk) is not None:
            if len(eod_flags) > 0:
                eod_flags[-1] = 1
    bot_answer_flags = bot_answer_flags[1:]
    # print('len(splitted[-1]) =', len(splitted[-1]))
    # print('len(speaker_flags) =', len(speaker_flags))
    if len(splitted[-1]) > 0:
        if splitted[-1][-1] == '\n' or speaker_flags[-1] != 0:
            bot_answer_flags.append(0)
        else:
            bot_answer_flags.append(0)
    else:
        bot_answer_flags.append(1)
    return [new_text, eod_flags, speaker_flags, bot_answer_flags, number_of_speakers]


class SimpleFontainBatcher(object):

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        tmp_output = process_input_text(text)
        [self._text, self._eod_flags, self._speaker_flags,
         self._bot_answer_flags, self._number_of_speakers] = tmp_output
        # print('self._speaker_flags:', self._speaker_flags[:5000])
        # print('self._eod_flags:', self._eod_flags[:5000])
        # print('self._bot_answer_flags:', self._bot_answer_flags[:5000])
        # print('self._text:', self._text[:5000])
        self._text_size = len(self._text)
        self._batch_size = batch_size
        self._vocabulary = vocabulary
        self._vocabulary_size = len(self._vocabulary)
        self._character_positions_in_vocabulary = get_positions_in_vocabulary(self._vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_inputs, _ = self._start_batch()
        print('self._number_of_speakers:', self._number_of_speakers)

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    # def _start_batch(self):
    #     batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
    #     for b in range(self._batch_size):
    #         batch[b, char2id('\n', self._character_positions_in_vocabulary)] = 1.0
    #     return batch

    def _start_batch(self):
        base = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
        bot_answer_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        speaker_flags = np.zeros(shape=(self._batch_size, self._number_of_speakers))
        eod_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        for b in range(self._batch_size):
            base[b, char2id('\n', self._character_positions_in_vocabulary)] = 1.0
            speaker_flags[b, 1] = 1.
            bot_answer_flags[b, 0] = 0.
        start_inputs = np.concatenate((base, speaker_flags, bot_answer_flags, eod_flags), 1)
        start_labels = np.concatenate((base, bot_answer_flags), 1)
        return start_inputs, start_labels

    def _zero_labels(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size + 1), dtype=np.float32)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        base = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
        bot_answer_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        speaker_flags = np.zeros(shape=(self._batch_size, self._number_of_speakers), dtype=np.float32)
        eod_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        for b in range(self._batch_size):
            base[b, char2id(self._text[self._cursor[b]], self._character_positions_in_vocabulary)] = 1.0
            speaker_flags[b, self._speaker_flags[self._cursor[b]]] = 1.0
            eod_flags[b, 0] = float(self._eod_flags[self._cursor[b]])
            bot_answer_flags[b, 0] = float(self._bot_answer_flags[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        inputs = np.concatenate((base, speaker_flags, bot_answer_flags, eod_flags), 1)
        labels = np.concatenate((base, bot_answer_flags), 1)
        return inputs, labels

    def char2vec(self, char, speaker_idx, eod):
        return np.stack(char2vec(self._character_positions_in_vocabulary,
                                 char,
                                 speaker_flag_size=self._num_of_speakers,
                                 speaker_idx=speaker_idx,
                                 eod=eod)), np.stack(self._zero_labels())

    def pred2vec(self, pred):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        return batch, self._zero_batch()

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        inputs = [self._last_inputs]
        labels = list()
        for step in range(self._num_unrollings):
            tmp_inputs, tmp_labels = self._next_batch()
            inputs.append(tmp_inputs)
            labels.append(tmp_labels)
        self._last_inputs = inputs[-1]
        return np.stack(inputs[:-1]), np.concatenate(labels, 0)


class SimpleFontain(Model):
    _name = 'simple_fontain'

    @staticmethod
    def get_special_args():
        return {'dialog_switch': True}

    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    def _layer(self,
               idx,
               state,
               bottom_up):

        with tf.name_scope('LSTM_layer_%s' % (idx)):
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
                                               [3 * self._num_nodes[idx], self._num_nodes[idx]],
                                               axis=1,
                                               name="split_to_function_arguments")

            gate_concat = tf.sigmoid(sigmoid_arg, name="gate_concat")
            [forget_gate, input_gate, output_gate] = tf.split(gate_concat,
                                                              3,
                                                              axis=1,
                                                              name="split_to_gates_op")
            modification_vector = tf.tanh(tanh_arg, name="modification_vector")

            prepaired_input = tf.multiply(modification_vector, input_gate, name="prepaired_input")
            prepaired_memory = tf.multiply(state[1], forget_gate, name="prepaired_memory")
            new_memory = tf.add(prepaired_memory, prepaired_input, name="new_memory")
            new_hidden = tf.multiply(tf.tanh(new_memory, name="tanh_result_for_new_hidden"),
                                     output_gate,
                                     name="new_hidden")
        return new_hidden, new_memory

    def _rnns(self, embedding, from_attention, state, iter_idx):
        # This function implements processing of one character embedding by HM_LSTM
        # 'inp' is one character embedding
        # 'state' is network state from previous layer
        # Method returns: new state of the network which includes hidden states,
        # memory states and boundary states for all layers; concatenated boundaries for all
        # layers ([batch_size, self._num_layers-1])

        with tf.name_scope('iteration_%s' % iter_idx):
            hidden = tf.concat([embedding, from_attention], 1)
            new_state = list()
            for idx in range(self._num_layers):
                hidden, memory = self._layer(idx,
                                             state[idx],
                                             hidden)
                new_state.append((hidden, memory))

            return new_state

    def _decide_force_or_sample(self,
                                inp,
                                predictions,
                                bot_answer_flags):
        with tf.name_scope('force_or_sample'):
            answer = tf.reduce_max(predictions, axis=1, keep_dims=True)
            answer = tf.to_float(tf.equal(predictions, answer))
            current_batch_size = answer.get_shape().as_list()[0]
            speaker_flags = tf.slice(inp, [0, self._vocabulary_size], [current_batch_size, 2], name='sliced_flags')
            answer = tf.concat([answer, speaker_flags], 1)
            delimiter_vec = char2vec(self._character_positions_in_vocabulary,
                                     self._replica_delimiter,
                                     speaker_flag_size=self._flag_size,
                                     speaker_idx=0,
                                     bot_answer_flag=1,
                                     eod=False)
            [delimiter_vec, _] = np.split(delimiter_vec, [self._vocabulary_size + self._flag_size], axis=1)
            input_is_not_delimiter = tf.not_equal(tf.reduce_sum(inp*delimiter_vec, axis=1, keep_dims=True), 1.)
            bot_answer_flags = tf.cast(bot_answer_flags, tf.bool)
            mask = tf.to_float(tf.logical_and(bot_answer_flags, input_is_not_delimiter), name='mask')
            return mask * answer + (1. - mask) * inp

    def _embed(self, inp):
        return tf.matmul(inp, self._embedding_matrix) + self._embedding_bias

    def _renew_attention_vec(self, hidden_states, for_attention, idx):
        with tf.name_scope('attention_%s' % idx):
            hidden_states = tf.concat(hidden_states, 1)
            for_computing = tf.stack(for_attention[-self._attention_visibility:])
            scalar_products = tf.einsum('ijk,jk->ji', for_computing, hidden_states)
            scores = tf.nn.softmax(scalar_products, name='scores')
        return tf.einsum('ijk,ji->jk', for_computing, scores)

    def _output_module(self,
                       hidden_states):
        # hidden_states is list of hidden_states by layer, concatenated along batch dimension
        with tf.name_scope('output_module'):
            concat = tf.concat(hidden_states, 1, name="total_concat_hidden")
            output_module_gates = tf.sigmoid(tf.matmul(concat,
                                                       self._output_gates_matrix,
                                                       name="matmul_in_output_gates_matrix"),
                                             name="output_gates_concat")
            output_module_gates = tf.split(output_module_gates,
                                           self._num_layers,
                                           axis=1,
                                           name="output_gates")
            gated_hidden_states = list()
            for idx, hidden_state in enumerate(hidden_states):
                gated_hidden_states.append(tf.multiply(output_module_gates[idx],
                                                       hidden_state,
                                                       name="gated_hidden_states_%s"%idx))
            gated_hidden_states = tf.concat(gated_hidden_states,
                                            1,
                                            name="gated_hidden_states")
            output_embeddings = tf.nn.relu(tf.add(tf.matmul(gated_hidden_states,
                                                            self._output_embedding_matrix,
                                                            name="matmul_in_output_embeddings"),
                                                  self._output_embedding_bias,
                                                  name="xW_plus_b_in_output_embeddings"),
                                           name="output_embeddings")
            logits = tf.add(tf.matmul(output_embeddings,
                                      self._output_matrix,
                                      name="matmul_in_logits"),
                            self._output_bias,
                            name="logits")
            return logits

    def _iter(self, inp, state, last_predictions, bot_answer_flags, from_attention, iter_idx, counter):
        with tf.name_scope('iter_%s' % iter_idx):
            inp = self._decide_force_or_sample(inp, last_predictions, bot_answer_flags)
            embedding = self._embed(inp)
            state = self._rnns(embedding, from_attention, state, iter_idx)
            only_hidden = [layer_state[0] for layer_state in state]
            logits = self._output_module(only_hidden)
        return [logits, state, tf.add(counter, 1, name='counter_%s' % iter_idx)]

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

    def _compose_reset_list(self, saved_last_sample_predictions, *args):
        with tf.name_scope('reset_list'):
            reset_list = list()
            name = self._extract_op_name(saved_last_sample_predictions.name)
            reset_list.append(tf.assign(saved_last_sample_predictions,
                                        np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                                                self._replica_delimiter),
                                                (1, 1)),
                                        name='assign_reset_%s' % name))
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                reset_list.append(tf.assign(variable, tf.zeros(shape), name='assign_reset_%s' % name))
            return reset_list

    @staticmethod
    def _partial_zeroing_out(eod_flags,
                             tensor):
        tensor_num_dims = len(tensor.get_shape().as_list())
        return tf.reshape((1. - eod_flags), [-1] + [1]*(tensor_num_dims - 1)) * tensor

    def _zeroing_out_after_end_of_dialog(self,
                                         eod_flags,
                                         state,
                                         for_attention,
                                         from_attention,
                                         list_of_predictions,
                                         iter_idx):
        with tf.name_scope('zeroing_out_%s' % iter_idx):
            new_state = list()
            for layer_state in state:
                new_layer_state = list()
                for component in layer_state:
                    new_layer_state.append(self._partial_zeroing_out(eod_flags, component))
                new_state.append(tuple(new_layer_state))
            if isinstance(for_attention, list):
                new_for_attention = list()
                for old in for_attention:
                    new_for_attention.append(self._partial_zeroing_out(eod_flags, old))
            else:
                new_for_attention = tf.transpose(
                    self._partial_zeroing_out(eod_flags, tf.transpose(for_attention, perm=[1,0, 2])),
                    perm=[1, 0, 2])
            new_from_attention = self._partial_zeroing_out(eod_flags, from_attention)
            new_list_of_predictions = list_of_predictions[:-1]
            new_list_of_predictions.append(self._partial_zeroing_out(eod_flags, list_of_predictions[-1]))
            return [new_state, new_for_attention, new_from_attention, new_list_of_predictions]

    def _unpack_sample_for_connections(self, for_connections):
        with tf.name_scope('unpacking_for_connections'):
            final = tf.unstack(for_connections)
            return final

    @staticmethod
    def _pack_sample_for_connections(for_connections):
        with tf.name_scope('packing_for_connections'):
            return tf.stack(for_connections)

    def __init__(self,
                 batch_size=64,
                 num_layers=3,
                 num_nodes=[20, 21, 24],
                 vocabulary_size=64,
                 embedding_size=47,
                 output_embedding_size=112,
                 flag_size=2,
                 attention_interval=3,
                 attention_visibility=3,
                 subsequence_length_in_intervals=7,
                 init_parameter=.3,
                 replica_delimiter='\n',
                 character_positions_in_vocabulary=None):
        self._batch_size = batch_size
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._output_embedding_size = output_embedding_size
        self._flag_size = flag_size
        self._attention_interval = attention_interval
        self._attention_visibility = attention_visibility
        self._subsequence_length_in_intervals = subsequence_length_in_intervals
        self._num_unrollings = self._subsequence_length_in_intervals * self._attention_interval
        self._init_parameter = init_parameter
        self._replica_delimiter = replica_delimiter
        self._character_positions_in_vocabulary = character_positions_in_vocabulary

        self.Matrices = list()
        self.Biases = list()

        # tensor name templates for HM_LSTM parameters
        init_matr_name = "LSTM_matrix_%s_initializer"
        init_bias_name = "LSTM_bias_%s_initializer"
        matr_name = "LSTM_matrix_%s"
        bias_name = "LSTM_bias_%s"

        input_dim = self._embedding_size + sum(self._num_nodes) + self._num_nodes[0]
        stddev = self._init_parameter * math.sqrt((1. / input_dim))
        self.Matrices.append(tf.Variable(tf.truncated_normal([input_dim,
                                                              4 * self._num_nodes[0]],
                                                             mean=0.,
                                                             stddev=stddev,
                                                             name=init_matr_name % 0),
                                         name=matr_name % 0))
        self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[0]],
                                                name=init_bias_name % 0),
                                       name=bias_name % 0))

        if self._num_layers > 1:
            for i in range(self._num_layers - 1):
                input_dim = self._num_nodes[i] + self._num_nodes[i+1]
                stddev = self._init_parameter * math.sqrt((1. / input_dim))
                self.Matrices.append(tf.Variable(tf.truncated_normal([input_dim,
                                                                      4 * self._num_nodes[i + 1]],
                                                                     mean=0.,
                                                                     stddev=stddev,
                                                                     name=init_matr_name % (i + 1)),
                                                 name=matr_name % (i + 1)))
                self.Biases.append(tf.Variable(tf.zeros([4 * self._num_nodes[i + 1]],
                                                        name=init_bias_name % (i + 1)),
                                               name=bias_name % (i + 1)))

        stddev = self._init_parameter * math.sqrt(1. / self._vocabulary_size + self._flag_size)
        self._embedding_matrix = tf.Variable(
            tf.truncated_normal([self._vocabulary_size + self._flag_size, self._embedding_size],
                                mean=0.,
                                stddev=stddev),
            name='embedding_matrix')
        self._embedding_bias = tf.Variable(tf.zeros([self._embedding_size]),
                                           name='embedding_bias')

        stddev = self._init_parameter * math.sqrt(1. / sum(self._num_nodes))
        self._output_embedding_matrix = tf.Variable(
            tf.truncated_normal([sum(self._num_nodes), self._output_embedding_size],
                                mean=0.,
                                stddev=stddev),
            name='output_embedding_matrix')
        self._output_embedding_bias = tf.Variable(tf.zeros([self._output_embedding_size]),
                                                  name='output_embedding_bias')

        stddev = self._init_parameter * math.sqrt(1. / sum(self._num_nodes))
        self._output_gates_matrix = tf.Variable(
            tf.truncated_normal([sum(self._num_nodes), 3],
                                mean=0.,
                                stddev=stddev),
            name='output_gates_matrix')

        stddev = self._init_parameter * math.sqrt(1. / self._output_embedding_size)
        self._output_matrix = tf.Variable(
            tf.truncated_normal([self._output_embedding_size, self._vocabulary_size],
                                mean=0.,
                                stddev=stddev),
            name='output_matrix')
        self._output_bias = tf.Variable(tf.zeros([self._vocabulary_size]),
                                        name='output_bias')

        saved_for_attention = list()
        with tf.name_scope('train'):
            saved_state = list()
            saved_state_templ = "saved_state_layer%s_number%s"
            for i in range(self._num_layers):
                saved_state.append((tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]]),
                                                trainable=False,
                                                name=saved_state_templ % (i, 0)),
                                    tf.Variable(tf.zeros([self._batch_size, self._num_nodes[i]]),
                                                trainable=False,
                                                name=saved_state_templ % (i, 1))))

            for idx in range(self._attention_visibility):
                saved_for_attention.append(tf.Variable(tf.zeros([self._batch_size, sum(self._num_nodes)]),
                                                       trainable=False,
                                                       name='saved_attention_%s' % idx))
            saved_last_predictions = tf.Variable(np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                                                         self._replica_delimiter),
                                                         (self._batch_size, 1)),
                                                 trainable=False,
                                                 name='saved_last_predictions')

            self.inputs = tf.placeholder(tf.float32, shape=[self._num_unrollings, self._batch_size,
                                                            self._vocabulary_size + self._flag_size + 2],
                                         name='inputs')
            self.labels = tf.placeholder(tf.float32, shape=[self._batch_size*self._num_unrollings,
                                                            self._vocabulary_size + 1],
                                         name='values')

            [inputs, bot_answer_flags, eod_flags] = tf.split(self.inputs,
                                                             [self._vocabulary_size + self._flag_size, 1, 1],
                                                             axis=2,
                                                             name='inputs_and_flags')

            inputs = tf.unstack(inputs)
            eod_flags = tf.unstack(eod_flags)
            [labels, _] = tf.split(self.labels, [self._vocabulary_size, 1], axis=1)
            bot_answer_flags_splitted = tf.unstack(bot_answer_flags)
            bot_answer_flags = tf.reshape(bot_answer_flags, [self._batch_size * self._num_unrollings])

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            for_attention = list(saved_for_attention)
            list_of_predictions = [saved_last_predictions]
            list_of_logits = list()

            state = saved_state
            counter = tf.constant(0, name='train_counter_0')
            for iter_idx, inp in enumerate(inputs):
                if iter_idx % self._attention_interval == 0:
                    from_attention = self._renew_attention_vec(
                        [layer_state[0] for layer_state in state], for_attention, iter_idx)
                tmp_output = self._zeroing_out_after_end_of_dialog(eod_flags[iter_idx],
                                                                   state,
                                                                   for_attention,
                                                                   from_attention,
                                                                   list_of_predictions,
                                                                   iter_idx)
                [state, for_attention, from_attention, list_of_predictions] = tmp_output
                [logits, state, counter] = self._iter(inp,
                                                      state,
                                                      list_of_predictions[-1],
                                                      bot_answer_flags_splitted[iter_idx],
                                                      from_attention,
                                                      iter_idx,
                                                      counter)
                if iter_idx % self._attention_interval == 0:
                    for_attention.append(tf.concat([layer_state[0] for layer_state in state], 1))
                list_of_logits.append(logits)
                predictions = tf.nn.softmax(logits, name='predictions_%s' % iter_idx)
                list_of_predictions.append(predictions)

            save_list = self._compose_save_list([saved_last_predictions, list_of_predictions[-1]],
                                                [saved_state, state])
            save_list.extend(self._compose_save_list_secure([saved_for_attention, for_attention]))
            logits = tf.concat(list_of_logits, 0, name='all_logits')
            logits = tf.reshape(bot_answer_flags, [self._batch_size * self._num_unrollings, 1]) * logits
            number_of_bot_characters = tf.reduce_sum(bot_answer_flags)
            there_is_bot_answers = tf.not_equal(number_of_bot_characters, 0.)
            regularizer = tf.contrib.layers.l2_regularizer(.5)

            l2_loss = regularizer(self._output_embedding_matrix)
            output_embedding_matrix_shape = self._output_embedding_matrix.get_shape().as_list()
            l2_divider = float(output_embedding_matrix_shape[0] * output_embedding_matrix_shape[1])

            with tf.control_dependencies(save_list):
                self.predictions = tf.nn.softmax(logits)
                ce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='softmax_cross_entropy')
                normal_loss = tf.reduce_sum(ce * bot_answer_flags) / (number_of_bot_characters + 1e-12)
                # normal_loss = tf.reduce_mean(
                #     tf.nn.softmax_cross_entropy_with_logits(
                #         labels=labels, logits=logits, name='softmax_cross_entropy'),
                #     name='normal_loss')
                self.loss = tf.cond(there_is_bot_answers,
                                    true_fn=lambda: normal_loss,
                                    false_fn=lambda: 0.)
            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss + l2_loss / l2_divider))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.)
            # print('Names of gradients:')
            # for grad in gradients:
            #     print(grad.name)
            self.train_op = optimizer.apply_gradients(zip(gradients, v))

        with tf.name_scope('validation'):
            saved_sample_counter = tf.Variable(0, trainable=False, name='valid_counter')
            saved_sample_state = list()
            saved_state_templ = "saved_sample_state_layer%s_number%s"
            for i in range(self._num_layers):
                saved_sample_state.append((tf.Variable(tf.zeros([1, self._num_nodes[i]]),
                                                       trainable=False,
                                                       name=saved_state_templ % (i, 0)),
                                           tf.Variable(tf.zeros([1, self._num_nodes[i]]),
                                                       trainable=False,
                                                       name=saved_state_templ % (i, 1))))

            saved_last_sample_predictions = tf.Variable(
                np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                        self._replica_delimiter),
                        (1, 1)),
                trainable=False,
                name='saved_last_sample_predictions')

            saved_sample_from_attention = tf.Variable(tf.zeros([1, sum(self._num_nodes)]), trainable=False)

            self.sample_inputs = tf.placeholder(tf.float32, shape=[1, 1,
                                                                   self._vocabulary_size + self._flag_size + 2])

            tmp_output = tf.split(self.sample_inputs,
                                  [self._vocabulary_size + self._flag_size, 1, 1],
                                  axis=2)
            [sample_inputs, sample_bot_answer_flags, sample_eod_flags] = tmp_output

            sample_inputs = tf.reshape(sample_inputs, [1, self._vocabulary_size + self._flag_size])
            sample_eod_flags = tf.reshape(sample_eod_flags, [1, 1])
            sample_bot_answer_flags = tf.reshape(sample_bot_answer_flags, [1, 1])


            it_is_time_to_reset_from_attention = tf.equal(tf.mod(saved_sample_counter, self._attention_interval), 0)


            saved_sample_for_attention = tf.Variable(tf.zeros([self._attention_visibility, 1, sum(self._num_nodes)]),
                                                     trainable=False,
                                                     name='saved_sample_state')
            tmp_output = self._zeroing_out_after_end_of_dialog(sample_eod_flags,
                                                               saved_sample_state,
                                                               saved_sample_for_attention,
                                                               saved_sample_from_attention,
                                                               [saved_last_sample_predictions],
                                                               0)
            [sample_state, sample_for_attention, sample_from_attention, list_of_sample_predictions] = tmp_output
            sample_hidden_states = [layer_state[0] for layer_state in saved_sample_state]
            sample_from_attention = tf.cond(it_is_time_to_reset_from_attention,
                                            true_fn=lambda: self._renew_attention_vec(sample_hidden_states,
                                                                                      sample_for_attention,
                                                                                      0),
                                            false_fn=lambda: sample_from_attention)

            sample_for_attention = self._unpack_sample_for_connections(saved_sample_for_attention)
            sample_for_attention = list(sample_for_attention)

            tmp_output = self._iter(sample_inputs,
                                    sample_state,
                                    list_of_sample_predictions[-1],
                                    sample_bot_answer_flags,
                                    sample_from_attention,
                                    0,
                                    saved_sample_counter)

            [sample_logits, sample_state, sample_counter] = tmp_output
            sample_for_attention.append(tf.concat(sample_hidden_states, 1))
            sample_for_attention = self._pack_sample_for_connections(sample_for_attention)
            sample_for_attention = tf.cond(
                it_is_time_to_reset_from_attention,
                true_fn=lambda: tf.split(sample_for_attention, [1, self._attention_visibility], axis=0)[1],
                false_fn=lambda: tf.split(sample_for_attention, [self._attention_visibility, 1], axis=0)[0]
            )
            sample_save_list = self._compose_save_list([saved_last_sample_predictions,
                                                        tf.nn.softmax(sample_logits)],
                                                       [saved_sample_state, sample_state],
                                                       [saved_sample_from_attention, sample_from_attention],
                                                       [saved_sample_counter, sample_counter])
            sample_save_list.extend(self._compose_save_list_secure([saved_sample_for_attention, sample_for_attention]))

            reset_list = self._compose_reset_list(saved_last_sample_predictions,
                                                  saved_sample_state,
                                                  saved_sample_for_attention,
                                                  saved_sample_from_attention)
            self.reset_sample_state = tf.group(*reset_list)
            with tf.control_dependencies(sample_save_list):
                self.sample_predictions = tf.nn.softmax(sample_logits)
        saved_var_list = list(self.Matrices)
        saved_var_list.extend(self.Biases)
        saved_var_list.append(self._embedding_matrix)
        saved_var_list.append(self._embedding_bias)
        saved_var_list.append(self._output_embedding_matrix)
        saved_var_list.append(self._output_embedding_bias)
        saved_var_list.append(self._output_gates_matrix)
        saved_var_list.append(self._output_matrix)
        saved_var_list.append(self._output_bias)
        self.saver = tf.train.Saver(saved_var_list, max_to_keep=None)

    def get_default_hooks(self):
        hooks = dict()
        hooks['inputs'] = self.inputs
        hooks['labels'] = self.labels
        hooks['train_op'] = self.train_op
        hooks['learning_rate'] = self.learning_rate
        hooks['loss'] = self.loss
        hooks['predictions'] = self.predictions
        hooks['validation_inputs'] = self.sample_inputs
        hooks['validation_predictions'] = self.sample_predictions
        hooks['reset_validation_state'] = self.reset_sample_state
        hooks['saver'] = self.saver
        return hooks

    @staticmethod
    def check_kwargs(**kwargs):
        pass














