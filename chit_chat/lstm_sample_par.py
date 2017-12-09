from __future__ import print_function
import re
import numpy as np
import tensorflow as tf
from some_useful_functions import (construct, create_vocabulary, char_2_base_vec,
                                   get_positions_in_vocabulary, pred2vec, vec2char,
                                   char2id, id2char, flatten, get_available_gpus, device_name_scope,
                                   average_gradients, average_gradients_not_balanced, get_num_gpus_and_bs_on_gpus)


url = 'http://mattmahoney.net/dc/'


def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size + speaker_flag_size + 1), dtype=np.float32)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    vec[0, voc_size + speaker_idx] = 1.0
    if speaker_idx > 0:
        vec[0, voc_size + speaker_flag_size] = 1.
    return vec


def process_input_text(text):
    # print('(process_input_text)len(text):', len(text))
    raw_splitted = text.split('<')
    splitted = list()
    for chunk in raw_splitted:
        if len(chunk) > 0:
            splitted.append(chunk)
    # print('(process_input_text)len(\'\'.join(splitted)):', len(''.join(splitted)))
    new_text = ''
    bot_speaks_flags = list()
    speaker_flags = list()
    flag = 0
    for chunk_idx, chunk in enumerate(splitted):
        match = re.match("[0-9]+в?>", chunk)
        if match is not None or chunk_idx == 0:
            if match is not None:
                span = match.span()
                string = chunk[span[1]:]
            else:
                # print('chunk_idx == 0, chunk =', chunk)
                string = chunk
            new_text += string
            length = len(string)
            bot_speaks_flags.extend([flag]*length)
            speaker_flags.extend([flag]*length)
            flag = (flag + 1) % 2

    return [new_text, speaker_flags, bot_speaks_flags]


def process_input_text_reg(text):
    interval = 5
    new_text = re.sub('<[^>]>', '', text)
    bot_speaks_flags = [(k // interval) % 2 for k in range(len(new_text))]
    speaker_flags = bot_speaks_flags
    return [new_text, speaker_flags, bot_speaks_flags]


class LstmBatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size)

    @staticmethod
    def pred2vec(pred, next_speaker_idx, speaker_flag_size):
        shape = pred.shape
        batch_size = shape[0]
        voc_size = shape[1]
        batch = np.zeros(shape=(batch_size, voc_size + speaker_flag_size + 1),
                         dtype=np.float32)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        batch[0, voc_size + next_speaker_idx] = 1.0
        if next_speaker_idx > 0:
            batch[0, voc_size + speaker_flag_size] = 1.0
        return batch

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):

        tmp_output = process_input_text_reg(text)
        # tmp_output = process_input_text(text)
        [self._text, self._speaker_flags, self._bot_speaks_flags] = tmp_output
        # print('self._speaker_flags:', self._speaker_flags[:5000])
        # print('self._bot_speaks_flags:', self._bot_speaks_flags[:5000])
        # print('self._text:', self._text[:5000])
        # print('(__init__)len(self._text):', len(self._text))
        # print('len(self._bot_speaks_flags):', len(self._bot_speaks_flags))
        # print('sum(self._bot_speaks_flags):', sum(self._bot_speaks_flags))
        self._text_size = len(self._text)
        self._batch_size = batch_size
        self._vocabulary = vocabulary
        self._vocabulary_size = len(self._vocabulary)
        self._number_of_speakers = 2
        self._character_positions_in_vocabulary = get_positions_in_vocabulary(self._vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_inputs, _ = self._start_batch()

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        base = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
        bot_speaks_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        speaker_flags = np.zeros(shape=(self._batch_size, self._number_of_speakers))
        for b in range(self._batch_size):
            base[b, char2id('\n', self._character_positions_in_vocabulary)] = 1.0
            speaker_flags[b, 1] = 1.
            bot_speaks_flags[b, 0] = 0.
        start_inputs = np.concatenate((base, speaker_flags, bot_speaks_flags), 1)
        start_labels = np.concatenate((base, bot_speaks_flags), 1)
        return start_inputs, start_labels

    def _zero_labels(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size + 1), dtype=np.float32)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        base = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float32)
        bot_speaks_flags = np.zeros(shape=(self._batch_size, 1), dtype=np.float32)
        speaker_flags = np.zeros(shape=(self._batch_size, self._number_of_speakers), dtype=np.float32)
        for b in range(self._batch_size):
            try:
                pos = self._cursor[b]
                chr = self._text[pos]
                chr_id = char2id(chr, self._character_positions_in_vocabulary)
                base[b, chr_id] = 1.0
            except IndexError:
                # print('(_next_batch)self._cursor:', self._cursor)
                # print('(_next_batch)b:', b)
                # print('(_next_batch)self._text:', self._text)
                # print('(_next_batch)pos:', pos)
                raise
            speaker_flags[b, self._speaker_flags[self._cursor[b]]] = 1.0
            bot_speaks_flags[b, 0] = float(self._bot_speaks_flags[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        inputs = np.concatenate((base, speaker_flags, bot_speaks_flags), 1)
        labels = np.concatenate((base, bot_speaks_flags), 1)
        return inputs, labels

    def char2batch(self, char):
        return np.stack(char2vec(char, self._character_positions_in_vocabulary)), np.stack(self._zero_labels())

    def pred2batch(self, pred):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        return np.stack([batch]), np.stack([self._zero_labels()])

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
        inputs = np.stack(inputs[:-1])
        labels = np.concatenate(labels, 0)
        _, bot_speaks_flags_in_inputs = np.split(inputs, [self._vocabulary_size + 2], axis=2)
        _, bot_speaks_flags_in_labels = np.split(labels, [self._vocabulary_size], axis=1)
        if len(self._text) < 1000:
            type = 'validation'
        else:
            type = 'train'
        # print('(%s)(LstmBatchGenerator.next)bot_speaks_flags_in_inputs:' % type, bot_speaks_flags_in_inputs)
        # print('(%s)(LstmBatchGenerator.next)bot_speaks_flags_in_labels:' % type, bot_speaks_flags_in_labels)
        # sum = np.sum(bot_speaks_flags_in_inputs)
        # print('(%s)(LstmBatchGenerator.next)number of ones:' % type, np.sum(bot_speaks_flags_in_inputs))
        # print('(%s)(LstmBatchGenerator.next)number of zeros:' % type, bot_speaks_flags_in_labels.shape[0] - sum)
        one_chunk_in, _ = np.split(bot_speaks_flags_in_inputs, [1], axis=1)
        one_chunk_in = np.reshape(one_chunk_in, [-1])
        bot_speaks_flags_in_labels = np.reshape(bot_speaks_flags_in_labels, (self._num_unrollings, self._batch_size, 1))
        one_chunk_out, _ = np.split(bot_speaks_flags_in_labels, [1], axis=1)
        one_chunk_out = np.reshape(one_chunk_out, [-1])
        one_chunk_flags = np.stack((one_chunk_in, one_chunk_out), axis=1)
        # print('(next) one_chunk_flags:', one_chunk_flags)
        return inputs, labels


def characters(probabilities, vocabulary):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c, vocabulary) for c in np.argmax(probabilities, 1)]


def batches2string(batches, vocabulary):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [""] * batches[0].shape[0]
    for b in batches:
        s = ["".join(x) for x in zip(s, characters(b, vocabulary))]
    return s


class Model(object):

    @classmethod
    def get_name(cls):
        return cls._name


class Lstm(Model):
    _name = 'lstm_sample'

    @classmethod
    def check_kwargs(cls,
                     **kwargs):
        pass

    @classmethod
    def get_name(cls):
        return cls._name

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

    def _lstm_layer(self, inp, state, layer_idx):
        with tf.name_scope('lstm_layer_%s' % layer_idx):
            matr = self._lstm_matrices[layer_idx]
            bias = self._lstm_biases[layer_idx]
            nn = self._num_nodes[layer_idx]
            x = tf.concat([tf.nn.dropout(inp, self.dropout_keep_prob), state[0]], 1, name='X')
            linear_res = tf.add(tf.matmul(x, matr, name='matmul'), bias, name='linear_res')
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3*nn, nn], axis=1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, (new_hidden_state, new_cell_state)

    def _rnn_iter(self, embedding, all_states):
        with tf.name_scope('rnn_iter'):
            new_all_states = list()
            output = embedding
            for layer_idx, state in enumerate(all_states):
                output, state = self._lstm_layer(output, state, layer_idx)
                new_all_states.append(state)
            return output, new_all_states

    def _rnn_module(self, embeddings, all_states):
        with tf.name_scope('rnn_iter'):
            new_all_states = list()
            output = embeddings
            for layer_idx, state in enumerate(all_states):
                output, state = self._lstm_layer(output, state, layer_idx)
                new_all_states.append(state)
            return output, new_all_states

    def _embed(self, inp):
        return tf.matmul(inp, self._embedding_matrix)

    def _output_module(self, rnn_output):
        with tf.name_scope('output_module'):
            hs = rnn_output
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

    def _generate_random(self, shape, one_prob):
        with tf.name_scope('generate_random'):
            un = tf.random_uniform(shape, maxval=1.)
            return tf.to_float(un < one_prob, name='random_modifier')

    def _decide_force_or_sample(self,
                                inp,
                                predictions,
                                in_s_flags):
        with tf.name_scope('force_or_sample'):
            answer = tf.reduce_max(predictions, axis=1, keep_dims=True, name='highest_scores_in_preds')
            answer = tf.to_float(tf.equal(predictions, answer, name='one_hot_mask_in_preds'), name='labels_of_predictions')
            current_batch_size = answer.get_shape().as_list()[0]
            speaker_flags = tf.slice(inp,
                                     [0, self._vocabulary_size],
                                     [current_batch_size, 2],
                                     name='sliced_flags')
            answer = tf.concat([answer, speaker_flags], 1, name='sampled_answer')
            delimiter_vec = char2vec('\n',
                                     self._character_positions_in_vocabulary,
                                     0,
                                     2)
            inp_to_comp = tf.slice(inp,
                                   [0, 0],
                                   [current_batch_size, self._vocabulary_size],
                                   name='input_to_comp_with_delim_vec')
            [delim_to_comp, _] = np.split(delimiter_vec, [self._vocabulary_size], axis=1)
            input_is_not_delimiter = tf.not_equal(
                tf.reduce_sum(inp_to_comp * delim_to_comp,
                              axis=1,
                              keep_dims=True),
                1.,
                name='input_is_not_delimiter')
            random_modifier = self._generate_random(in_s_flags.get_shape().as_list(), self.sampling_prob)
            in_s_flags = tf.multiply(in_s_flags, random_modifier, name='in_s_flags_after_modification')
            in_s_flags = tf.cast(in_s_flags, tf.bool, name='in_s_flags_am_bool')
            mask = tf.to_float(tf.logical_and(in_s_flags, input_is_not_delimiter),
                               name='mask')
            return tf.stop_gradient(tf.add(mask * answer, (1. - mask) * inp), name='inp_after_choosing')
            # return tf.stop_gradient(tf.add((1. - mask) * answer, mask * inp), name='inp_after_choosing')

    def _iter(self, inp, all_states, last_predictions, in_s_flags, iter_idx):
        with tf.name_scope('iter_%s' % iter_idx):
            inp = self._decide_force_or_sample(inp, last_predictions, in_s_flags)
            embedding = self._embed(inp)
            with tf.name_scope('attaching_big_flag'):
                _, flag = tf.split(inp,
                                   [self._vocabulary_size, self._flag_size],
                                   axis=1,
                                   name='flag')
                first_speaker = tf.constant([[1., 0.]])
                comp_res = tf.stop_gradient(
                    tf.reduce_prod(
                        tf.to_float(tf.equal(flag, first_speaker)),
                        axis=1,
                        keep_dims=True,
                        name='comp_res'))
                big_flag = tf.tile((comp_res - .5) * 2., [1, self._big_size], name='big_flag')
                # with tf.device('/cpu:0'):
                #     big_flag = tf.Print(big_flag, [big_flag], message='big_flag')
                embedding = tf.concat([embedding, big_flag], 1, name='flagged_embedding')
            output, all_states = self._rnn_module(embedding, all_states)
            logits = self._output_module(output)
        return [logits, all_states]

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

    def _compose_reset_list(self, *args):
        with tf.name_scope('reset_list'):
            reset_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                reset_list.append(tf.assign(variable, tf.zeros(shape), name='assign_reset_%s' % name))
            return reset_list

    def _compose_randomize_list(self, *args):
        with tf.name_scope('randomize_list'):
            randomize_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                assign_tensor = tf.truncated_normal(shape, stddev=1.)
                #assign_tensor = tf.Print(assign_tensor, [assign_tensor], message='assign tensor:')
                assign = tf.assign(variable, assign_tensor, name='assign_reset_%s' % name)
                randomize_list.append(assign)
            return randomize_list

    def _compute_lstm_matrix_parameters(self, idx):
        if idx == 0:
            # print(self._num_nodes)
            input_dim = self._num_nodes[0] + self._embedding_size + self._big_size
        else:
            input_dim = self._num_nodes[idx-1] + self._num_nodes[idx]
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
        num_active = list()
        with tf.name_scope('train'):
            for gpu_batch_size, gpu_name, device_inputs, device_labels, dev_in_s_flags, dev_out_s_flags in zip(
                    self._batch_sizes_on_gpus, self._gpu_names, self._inputs_by_device,
                    self._labels_by_device, self._in_s_flags_by_device, self._out_s_flags_by_device):
                with tf.device(gpu_name):
                    with tf.name_scope(device_name_scope(gpu_name)):
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

                        saved_last_predictions = tf.Variable(
                            np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                                    '\n'),
                                    (gpu_batch_size, 1)),
                            trainable=False,
                            name='saved_last_predictions')
                        list_of_predictions = [saved_last_predictions]
                        list_of_logits = list()
                        all_states = saved_states
                        for iter_idx, inp in enumerate(device_inputs):
                            [logits, all_states] = self._iter(inp, all_states, list_of_predictions[-1],
                                                              dev_in_s_flags[iter_idx], iter_idx)
                            list_of_logits.append(logits)
                            predictions = tf.nn.softmax(logits, name='predictions_%s' % iter_idx)
                            list_of_predictions.append(predictions)

                        save_ops = self._compose_save_list((saved_states, all_states),
                                                           (saved_last_predictions, list_of_predictions[-1]))
                        logits = tf.concat(list_of_logits, 0, name='all_logits')

                        with tf.control_dependencies(save_ops):
                            with tf.name_scope('device_gradient_computation'):
                                all_matrices = [self._embedding_matrix]
                                all_matrices.extend(self._lstm_matrices)
                                all_matrices.extend(self._output_matrices)
                                l2_loss = self._l2_loss(all_matrices)
                                random_modifier = self._generate_random(dev_out_s_flags.get_shape().as_list(),
                                                                        (1. - self.loss_comp_prob))
                                dev_out_s_flags = tf.stop_gradient(
                                    tf.subtract(1.,
                                                (1. - dev_out_s_flags) * random_modifier),
                                    name='final_dev_out_s_flags')
                                dev_out_s_flags = tf.reshape(dev_out_s_flags, [-1], name='dev_out_s_flags_reshaped')
                                number_of_considered_losses = tf.reduce_sum(dev_out_s_flags,
                                                                            name='number_of_considered_losses')
                                # ce = tf.reduce_mean(
                                #     tf.nn.softmax_cross_entropy_with_logits(labels=device_labels, logits=logits))
                                # loss = tf.reduce_sum(ce * dev_out_s_flags) / (number_of_considered_losses + 1e-12)
                                # losses.append(loss)

                                num_active.append(number_of_considered_losses)
                                ce = tf.nn.softmax_cross_entropy_with_logits(
                                    labels=device_labels,
                                    logits=logits,
                                    name='ce_not_filtered')
                                loss = tf.reduce_sum(ce * dev_out_s_flags, name='loss')
                                losses.append(loss)

                                # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                                grads_and_vars = optimizer.compute_gradients(loss + l2_loss)
                                tower_grads.append(grads_and_vars)

                                # splitting concatenated results for different characters
                                concat_pred = tf.nn.softmax(logits)
                                preds.append(tf.split(concat_pred, self._num_unrollings))

            with tf.device('/cpu:0'):
                with tf.name_scope(device_name_scope('/cpu:0') + '_gradients'):
                    # grads_and_vars = average_gradients(tower_grads)
                    grads_and_vars = average_gradients_not_balanced(tower_grads, num_active)

                    grads, v = zip(*grads_and_vars)
                    grads, _ = tf.clip_by_global_norm(grads, 1.)
                    # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    self.train_op = optimizer.apply_gradients(zip(grads, v))
                    self._hooks['train_op'] = self.train_op

                    # composing predictions
                    preds_by_char = list()
                    # print('preds:', preds)
                    for char_idx, one_char_preds in enumerate(zip(*preds)):
                        # print('one_char_preds:', one_char_preds)
                        preds_by_char.append(
                            tf.concat(
                                one_char_preds, 0,
                                name='one_char_preds_%s' % char_idx))
                    # print('len(preds_by_char):', len(preds_by_char))
                    self.predictions = tf.concat(preds_by_char, 0, name='predictions')
                    self._hooks['predictions'] = self.predictions

                    # print('self.predictions.get_shape().as_list():', self.predictions.get_shape().as_list())
                    # l = 0
                    # for loss, gpu_batch_size in zip(losses, self._batch_sizes_on_gpus):
                    #     l += float(gpu_batch_size) / float(self._batch_size) * loss

                    l = tf.identity(sum(losses), name='sum_of_all_losses')
                    num_active = tf.identity(sum(num_active), name='number_of_computed_losses')
                    # num_active = tf.Print(num_active, [l, num_active], message='loss_sum, num_active:')
                    self.loss = tf.divide(l, (num_active + 1e-12), name='average_loss')
                    self._hooks['loss'] = self.loss

    def _validation_graph(self):
        with tf.device(self._gpu_names[0]):
            with tf.name_scope('validation'):
                self.sample_input = tf.placeholder(tf.float32,
                                                   shape=[1, 1, self._vocabulary_size + 3],
                                                   name='sample_input')
                self._hooks['validation_inputs'] = self.sample_input

                sample_input, in_s_flag = tf.split(self.sample_input,
                                                   [self._vocabulary_size + 2, 1],
                                                   axis=2,
                                                   name='sample_input_and_in_s_flag')
                sample_input = tf.reshape(sample_input, [1, -1])
                in_s_flag = tf.reshape(in_s_flag, [1, 1])
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
                saved_sample_last_predictions = tf.Variable(
                    np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                            '\n'),
                            (1, 1)),
                    trainable=False,
                    name='saved_last_predictions')
                reset_list = self._compose_reset_list(saved_sample_state)
                reset_list.append(tf.assign(saved_sample_last_predictions,
                                            np.tile(char_2_base_vec(self._character_positions_in_vocabulary,
                                                                    '\n'),
                                                    (1, 1))
                                            ))

                self.reset_sample_state = tf.group(*reset_list)
                self._hooks['reset_validation_state'] = self.reset_sample_state

                randomize_list = self._compose_randomize_list(saved_sample_state)
                random_pred = tf.nn.softmax(tf.random_normal([1, self._vocabulary_size]))
                randomize_list.append(tf.assign(saved_sample_last_predictions, random_pred))
                self.randomize = tf.group(*randomize_list)
                self._hooks['randomize_sample_state'] = self.randomize

                [sample_logit, sample_state] = self._iter(sample_input, saved_sample_state,
                                                          saved_sample_last_predictions, in_s_flag, 0)

                sample_save_ops = self._compose_save_list([saved_sample_last_predictions,
                                                          tf.nn.softmax(sample_logit)],
                                                          [saved_sample_state, sample_state])

                with tf.control_dependencies(sample_save_ops):
                    self.sample_prediction = tf.nn.softmax(sample_logit)
                    self._hooks['validation_predictions'] = self.sample_prediction

    def __init__(self,
                 batch_size=64,
                 num_layers=2,
                 num_nodes=[112, 113],
                 num_output_layers=1,
                 num_output_nodes=[],
                 flag_size=2,
                 vocabulary_size=None,
                 embedding_size=128,
                 num_unrollings=10,
                 init_parameter=3.,
                 num_gpus=1,
                 regularization_rate=.000003,
                 character_positions_in_vocabulary=None,
                 regime='train'):

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
                           sampling_prob=None,
                           loss_comp_prob=None,
                           saver=None)

        self._batch_size = batch_size
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._num_output_layers = num_output_layers
        self._num_output_nodes = num_output_nodes
        self._flag_size = flag_size
        self._num_unrollings = num_unrollings
        self._init_parameter = init_parameter
        self._regularization_rate = regularization_rate
        self._big_size = 10

        self._gpu_names = get_available_gpus()
        num_available_gpus = len(self._gpu_names)
        num_gpus, self._batch_sizes_on_gpus = get_num_gpus_and_bs_on_gpus(self._batch_size, num_gpus, num_available_gpus)
        self._num_gpus = num_gpus

        self._character_positions_in_vocabulary = character_positions_in_vocabulary

        with tf.device('/cpu:0'):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            self.sampling_prob = tf.placeholder(tf.float32, name='sampling_prob')
            self.loss_comp_prob = tf.placeholder(tf.float32, name='loss_computation_prob')

            self.inputs = tf.placeholder(tf.float32,
                                         shape=[self._num_unrollings, self._batch_size, self._vocabulary_size + 3])
            self.labels = tf.placeholder(tf.float32,
                                         shape=[self._num_unrollings * self._batch_size, self._vocabulary_size + 1])


            #in_flags
            self._hooks['dropout'] = self.dropout_keep_prob
            self._hooks['sampling_prob'] = self.sampling_prob
            self._hooks['loss_comp_prob'] = self.loss_comp_prob
            self._hooks['inputs'] = self.inputs
            self._hooks['labels'] = self.labels

            inputs, in_s_flags = tf.split(self.inputs,
                                          [self._vocabulary_size + 2, 1],
                                          axis=2)

            labels, out_s_flags = tf.split(self.labels, [self._vocabulary_size, 1], axis=1)

            inputs = tf.split(inputs, self._batch_sizes_on_gpus, 1)
            self._inputs_by_device = list()
            for dev_idx, device_inputs in enumerate(inputs):
                self._inputs_by_device.append(tf.unstack(device_inputs, name='inp_on_dev_%s' % dev_idx))

            in_s_flags = tf.split(in_s_flags, self._batch_sizes_on_gpus, 1)
            self._in_s_flags_by_device = list()
            for dev_idx, device_in_s_flags in enumerate(in_s_flags):
                self._in_s_flags_by_device.append(tf.unstack(device_in_s_flags, name='in_s_flags_on_dev_%s' % dev_idx))

            labels = tf.reshape(labels, shape=(self._num_unrollings, self._batch_size, self._vocabulary_size))
            labels = tf.split(labels, self._batch_sizes_on_gpus, 1)
            self._labels_by_device = list()
            for dev_idx, device_labels in enumerate(labels):
                self._labels_by_device.append(
                    tf.reshape(device_labels,
                               [-1, self._vocabulary_size],
                               name='labels_on_dev_%s' % dev_idx))

            out_s_flags = tf.reshape(out_s_flags,
                                     shape=(self._num_unrollings, self._batch_size, 1))
            out_s_flags = tf.split(out_s_flags, self._batch_sizes_on_gpus, 1)
            self._out_s_flags_by_device = list()
            for dev_idx, device_out_s_flags in enumerate(out_s_flags):
                self._out_s_flags_by_device.append(
                    tf.reshape(device_out_s_flags,
                               [-1, 1],
                               name='out_s_on_dev_%s' % dev_idx))

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self.learning_rate

            self._embedding_matrix = tf.Variable(
                tf.truncated_normal([self._vocabulary_size + 2, self._embedding_size],
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
            saved_vars = dict()
            saved_vars['embedding_matrix'] = self._embedding_matrix
            for layer_idx, (lstm_matrix, lstm_bias) in enumerate(zip(self._lstm_matrices, self._lstm_biases)):
                saved_vars['lstm_matrix_%s' % layer_idx] = lstm_matrix
                saved_vars['lstm_bias_%s' % layer_idx] = lstm_bias
            for layer_idx, (output_matrix, output_bias) in enumerate(zip(self._output_matrices, self._output_biases)):
                saved_vars['output_matrix_%s' % layer_idx] = output_matrix
                saved_vars['output_bias_%s' % layer_idx] = output_bias
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
