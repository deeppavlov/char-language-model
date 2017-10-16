from __future__ import print_function
import numpy as np
import tensorflow as tf
import zipfile
import codecs
import os
from some_useful_functions import construct, create_vocabulary, get_positions_in_vocabulary, char2vec, char2id, id2char


url = 'http://mattmahoney.net/dc/'


class BatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._vocabulary = vocabulary
        self._vocabulary_size = len(self._vocabulary)
        self._characters_positions_in_vocabulary = get_positions_in_vocabulary(self._vocabulary)
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
            batch[b, char2id('\n', self._characters_positions_in_vocabulary)] = 1.0
        return batch

    def _zero_batch(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]], self._characters_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def char2vec(self, char):
        return np.stack(char2vec(char, self._characters_positions_in_vocabulary)), np.stack(self._zero_batch())

    def pred2vec(self, pred):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        return batch, self._zero_batch()

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
        return cls.name


class Vanilla(Model):
    _name = 'vanilla'

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

    def _iter(self, inp, hidden_state):
        X = tf.concat([inp, hidden_state], 1)
        output = tf.tanh(tf.matmul(X, self._weights) + self._bias)
        return output, output

    @staticmethod
    def form_list_of_kwargs(kwargs_for_building, build_hyperparameters):
        output = [(construct(kwargs_for_building), dict(), list())]
        lengths = list()
        for name, values in build_hyperparameters.items():
            new_output = list()
            lengths.append(len(values))
            for base in output:
                for idx, value in enumerate(values):
                    new_base = construct(base)
                    new_base[0][name] = value
                    new_base[1][name] = value
                    new_base[2].append(idx)
                    new_output.append(new_base)
            output = new_output
        sorting_factors = [1]
        for length in reversed(lengths[1:]):
            sorting_factors.append(sorting_factors[-1] * length)
        output = sorted(output,
                        key=lambda set: sum(
                            [point_idx*sorting_factor \
                             for point_idx, sorting_factor in zip(reversed(set[2][1:]), sorting_factors)]))
        return output

    def __init__(self,
                 batch_size=64,
                 num_nodes=64,
                 vocabulary_size=None,
                 num_unrollings=10):
        self._batch_size = batch_size
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._num_unrollings = num_unrollings

        self._weights = tf.Variable(tf.truncated_normal([self._vocabulary_size + self._num_nodes, self._num_nodes],
                                                        mean=0.,
                                                        stddev=.1))
        self._bias = tf.Variable(tf.zeros([self._num_nodes]))
        output_weights = tf.Variable(tf.truncated_normal([self._num_nodes, self._vocabulary_size],
                                                         mean=0.,
                                                         stddev=.1))
        output_bias = tf.Variable(tf.zeros([self._vocabulary_size]))

        saved_state = tf.Variable(tf.zeros([self._batch_size, self._num_nodes]), trainable=False)

        self.inputs = tf.placeholder(tf.float32, shape=[self._num_unrollings, self._batch_size, self._vocabulary_size])
        self.labels = tf.placeholder(tf.float32, shape=[self._num_unrollings * self._batch_size, self._vocabulary_size])

        inputs = tf.unstack(self.inputs)

        outputs = list()
        hidden_state = saved_state
        for inp in inputs:
            output, hidden_state = self._iter(inp, hidden_state)
            outputs.append(output)

        outputs = tf.concat(outputs, 0)
        save_ops = [tf.assign(saved_state, hidden_state)]
        with tf.control_dependencies(save_ops):
            logits = tf.matmul(outputs, output_weights) + output_bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        self.learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        self.train_op = optimizer.apply_gradients(zip(gradients, v))

        self.predictions = tf.nn.softmax(logits)

        self.sample_input = tf.placeholder(tf.float32, shape=[1, 1, self._vocabulary_size])
        sample_input = tf.reshape(self.sample_input, [1, -1])
        saved_sample_state = tf.Variable(tf.zeros([1, self._num_nodes]))
        self.reset_sample_state = tf.assign(saved_sample_state, tf.zeros([1, self._num_nodes]))
        sample_output, new_state = self._iter(sample_input, saved_sample_state)
        sample_save_ops = [tf.assign(saved_sample_state, new_state)]
        sample_logits = tf.matmul(sample_output, output_weights) + output_bias
        with tf.control_dependencies(sample_save_ops):
            self.sample_prediction = tf.nn.softmax(sample_logits)
        self.saver = tf.train.Saver(max_to_keep=None)

    def get_default_hooks(self):
        hooks = dict()
        hooks['inputs'] = self.inputs
        hooks['labels'] = self.labels
        hooks['train_op'] = self.train_op
        hooks['learning_rate'] = self.learning_rate
        hooks['loss'] = self.loss
        hooks['predictions'] = self.predictions
        hooks['validation_inputs'] = self.sample_input
        hooks['validation_predictions'] = self.sample_prediction
        hooks['reset_validation_state'] = self.reset_sample_state
        hooks['saver'] = self.saver
        return hooks

    def get_building_parameters(self):
        pass
