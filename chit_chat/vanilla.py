# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import random
import string
import tensorflow as tf
import tensorflow.python.ops.rnn_cell
import zipfile
#from six.moves import range
#from six.moves.urllib.request import urlretrieve
import collections
import matplotlib
import matplotlib.pyplot as plt
import codecs
import time
import os
import gc
#from six.moves import cPickle as pickle
from tensorflow.python import debug as tf_debug

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    # Download a file if not present, and make sure it's the right size.
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    if not os.path.exists('enwik8'):
        f = zipfile.ZipFile(filename)
        for name in f.namelist():
            full_text = tf.compat.as_str(f.read(name))
        f.close()
        """f = open('enwik8', 'w')
        f.write(text.encode('utf8'))
        f.close()"""
    else:
        f = open('enwik8', 'r')
        full_text = f.read().decode('utf8')
        f.close()
    return full_text

    f = codecs.open('enwik8', encoding='utf-8')
    text = f.read()
    f.close()
    return text


def check_not_one_byte(text):
    not_one_byte_counter = 0
    max_character_order_index = 0
    min_character_order_index = 2 ** 16
    present_characters = [0] * 256
    number_of_characters = 0
    for char in text:
        if ord(char) > 255:
            not_one_byte_counter += 1
        if len(present_characters) <= ord(char):
            present_characters.extend([0] * (ord(char) - len(present_characters) + 1))
            present_characters[ord(char)] = 1
            number_of_characters += 1
        elif present_characters[ord(char)] == 0:
            present_characters[ord(char)] = 1
            number_of_characters += 1
        if ord(char) > max_character_order_index:
            max_character_order_index = ord(char)
        if ord(char) < min_character_order_index:
            min_character_order_index = ord(char)
    return not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters


def create_vocabulary(text):
    all_characters = list()
    for char in text:
        if char not in all_characters:
            all_characters.append(char)
    return sorted(all_characters, key=lambda dot: ord(dot))


def get_positions_in_vocabulary(vocabulary):
    characters_positions_in_vocabulary = dict()
    for idx, char in enumerate(vocabulary):
        characters_positions_in_vocabulary[char] = idx
    return characters_positions_in_vocabulary


def filter_text(text, allowed_letters):
    new_text = ""
    for char in text:
        if char in allowed_letters:
            new_text += char
    return new_text


def char2id(char, characters_positions_in_vocabulary):
    if char in characters_positions_in_vocabulary:
        return characters_positions_in_vocabulary[char]
    else:
        print(u'Unexpected character: %s\nUnexpected character number: %s\n' % (char, ord(char)))
        return None


def char2vec(char, characters_positions_in_vocabulary):
    voc_size = len(characters_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float)
    vec[0, char2id(char, characters_positions_in_vocabulary)] = 1.0
    return vec


def id2char(dictid, vocabulary):
    voc_size = len(vocabulary)
    if (dictid >= 0) and (dictid < voc_size):
        return vocabulary[dictid]
    else:
        print(u"unexpected id")
        return u'\0'


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
