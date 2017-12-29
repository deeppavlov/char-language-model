import numpy as np
from some_useful_functions import char2vec, pred2vec, vec2char, get_positions_in_vocabulary, char2id, id2char
import re
NUMBER_OF_CHARS_IN_NGRAMS = 2


def custom_split(text, interval):
    res = list()
    counter = 0
    chunk = ''
    for c in text:
        if counter == interval:
            res.append(chunk)
            counter = 0
            chunk = ''
        chunk += c
        counter += 1
    if counter > 0:
        res.append(chunk)
    return res


def special_split(text):
    replicas = re.split('(\n)', text)
    pairs = list()
    for r in replicas:
        if len(r) > 0 and r != '\n':
            splitted = custom_split(r, NUMBER_OF_CHARS_IN_NGRAMS)
            for s in splitted[:-1]:
                pairs.append(s)

            try:
                if len(splitted[-1]) < NUMBER_OF_CHARS_IN_NGRAMS:
                    pairs.append(splitted[-1] + ' ' * (NUMBER_OF_CHARS_IN_NGRAMS - len(splitted[-1])))
                else:
                    pairs.append(splitted[-1])
            except IndexError:
                print('(special_split)splitted:', splitted)
                print('(special_split)replicas:', replicas)
                # print('(special_split)replicas:', replicas)
                raise

        else:
            pairs.append('\n')
    return pairs


def create_vocabulary(text):
    vocabulary = list()
    for offset in range(NUMBER_OF_CHARS_IN_NGRAMS):
        segments = special_split(text)
        for s in segments:
            if s not in vocabulary and len(s) > 0:
                vocabulary.append(s)
    return sorted(vocabulary)


class NgramsBatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(char2vec(char, character_positions_in_vocabulary), (1, 1, -1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec(pred), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def make_pairs(text, batch_gen_args):
        return special_split(text)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._pairs = self.make_pairs(self._text, None)
        self._number_of_pairs = len(self._pairs)
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._number_of_pairs // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id('\n', self.character_positions_in_vocabulary)] = 1.0
        return batch

    def _zero_batch(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            batch[b, char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return batch

    def char2batch(self, char):
        return np.stack(char2vec(char, self.character_positions_in_vocabulary)), np.stack(self._zero_batch())

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

    def _next_batch_with_tokens(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        tokens = list()
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            tokens.append(self._pairs[self._cursor[b]])
            batch[b, char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return batch, tokens

    def next_with_tokens(self):
        batches = [self._last_batch]
        batch, tokens = self._next_batch_with_tokens()
        batches.append(batch)
        self._last_batch = batches[-1]
        # print('(BpeBatchGenerator.next_with_tokens)tokens:', tokens)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0), tokens

