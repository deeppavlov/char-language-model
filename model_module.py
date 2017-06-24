# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import random
import string
import tensorflow as tf
import tensorflow.python.ops.rnn_cell 
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

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  #Download a file if not present, and make sure it's the right size.
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
    min_character_order_index = 2**16 
    present_characters = [0]*256
    number_of_characters = 0
    for i in range(len(text)):
        if ord(text[i]) > 255:
            not_one_byte_counter += 1 
        if len(present_characters) <  ord(text[i]):
            present_characters.extend([0]*(ord(text[i]) - len(present_characters) + 1))
            present_characters[ord(text[i])] = 1
            number_of_characters += 1
        elif present_characters[ord(text[i])] == 0:
            present_characters[ord(text[i])] = 1
            number_of_characters += 1
        if ord(text[i]) > max_character_order_index:
            max_character_order_index = ord(text[i])
        if ord(text[i]) < min_character_order_index:
            min_character_order_index = ord(text[i])
    return not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters


def char2id(char, characters_positions_in_vocabulary):
  if characters_positions_in_vocabulary[ord(char)] != -1:
    return characters_positions_in_vocabulary[ord(char)]
  else:
    print(u'Unexpected character: %s\nUnexpected character number: %s\nUnexpected character has its place = %s\n' % (char, ord(char), present_characters_indices[i]))
    return characters_positions_in_vocabulary[ord(char)]

def id2char(dictid, vocabulary):
  voc_size = len(vocabulary)
  if (dictid >= 0) and (dictid < voc_size):
    return vocabulary[dictid]
  else:
    print(u"unexpected id")
    return u'\0'


class BatchGenerator(object):
  def __init__(self, text, batch_size, vocabulary_size, characters_positions_in_vocabulary,  num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._vocabulary_size = vocabulary_size
    self._characters_positions_in_vocabulary = characters_positions_in_vocabulary
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]], self._characters_positions_in_vocabulary)] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

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


def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction, vocabulary_size):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution(vocabulary_size):
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

def percent_of_correct_predictions(predictions, labels):
    num_characters = predictions.shape[0]
    num_correct = 0
    for i in range(num_characters):
        if labels[i, np.argmax(predictions, axis=1)[i]] == 1:
            num_correct += 1
    return float(num_correct) / num_characters * 100



class MODEL(object):
    _train_batches = None
    _valid_batches = None
    SKIP_LENGTH = 100
                    
    def learn(self,
              session,
              min_num_steps,
              loss_frequency, # loss is calculated with frequency loss_frequency
              block_of_steps, #learning has a chance to be stopped after every block of steps
              num_stairs,
              decay,
              stop_percent,
              optional_feed_dict=None,
              half_life_fixed=False,
              fixed_num_steps=False):
        if not half_life_fixed and (self._last_num_steps > min_num_steps):
            min_num_steps = self._last_num_steps
        half_life = min_num_steps / num_stairs
            
            
        if self._train_batches is None:
            self._train_batches = BatchGenerator(self._train_text,
                                                 self._batch_size,
                                                 self._vocabulary_size,
                                                 self._characters_positions_in_vocabulary,
                                                 self._num_unrollings)
            
        losses = [0.]

        tf.initialize_all_variables().run()
        start_time = time.clock()
        mean_loss = 0.
        step = 0
        if not fixed_num_steps:
            stop_condition = ((step - 1) % block_of_steps == 0 and
                                (losses[len(losses) / 2] - losses[-1]) < losses[len(losses) / 2] * stop_percent / 100 and
                                step >= min_num_steps)
        else:
            stop_condition = (step >= min_num_steps)
        while not stop_condition:
            batches = self._train_batches.next()
            feed_dict = {self._half_life: half_life, self._decay: decay}
            if optional_feed_dict is not None:
                new_dict = dict()
                for optional_key in optional_feed_dict.keys():
                    exec("new_dict[%s] = %s" % (optional_key, optional_feed_dict[optional_key]))
                feed_dict.update(new_dict)
            for i in range(self._num_unrollings + 1):
                feed_dict[self._train_data[i]] = batches[i]
            _, l = session.run([self._optimizer,
                                self._loss],
                               feed_dict=feed_dict)
            mean_loss += l              
            if step % loss_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / loss_frequency
                    # The mean loss is an estimate of the loss over the last few batches.
                losses.append(mean_loss)
                mean_loss = 0
            step += 1
            if not fixed_num_steps:
                stop_condition = ((step - 1) % block_of_steps == 0 and
                                    (losses[len(losses) / 2] - losses[-1]) < losses[len(losses) / 2] * stop_percent / 100 and
                                    step >= min_num_steps)
            else:
                stop_condition = (step >= min_num_steps)
        finish_time = time.clock() 
        self._last_num_steps = step
        return finish_time - start_time
       
    def calculate_percentages(self,
                              session,
                              num_averaging_iterations):
        if not isinstance(self._valid_text, dict):
            data_for_plot = {'train': {'step': list(), 'percentage': list()},
                             'validation': {'step': list(), 'percentage': list()}}
        else:
            keys = self._valid_text.keys()
            data_for_plot = {'train': {'step': list(), 'percentage': list()}} 
            for key in keys:
                data_for_plot[key] = {'step': list(), 'percentage': list()}           
        
        for key in data_for_plot.keys():
            data_for_plot[key]['step'].append(-1)
        
        if self._valid_batches is None:
            if not isinstance(self._valid_text, dict): 
                self._valid_batches = BatchGenerator(self._valid_text,
                                                     1,
                                                     self._vocabulary_size,
                                                     self._characters_positions_in_vocabulary,
                                                     1)
            else:
                self._valid_batches = dict()
                for key in keys:
                    self._valid_batches[key] = BatchGenerator(self._valid_text[key],
                                                              1,
                                                              self._vocabulary_size,
                                                              self._characters_positions_in_vocabulary,
                                                              1)                    
        
        average_percentage_of_correct = 0.
        for _ in range(num_averaging_iterations):
            for _ in range(self.SKIP_LENGTH / self._num_unrollings):
                batches = self._train_batches.next()
                feed_dict = dict()
                for i in range(self._num_unrollings + 1):
                    feed_dict[self._train_data[i]] = batches[i]
                session.run(self._skip_operation, feed_dict=feed_dict)
            batches = self._train_batches.next()
            feed_dict = dict()
            for i in range(self._num_unrollings + 1):
                feed_dict[self._train_data[i]] = batches[i]            
            predictions = session.run(self._train_prediction, feed_dict=feed_dict)
            labels = np.concatenate(list(batches)[1:])
            average_percentage_of_correct += percent_of_correct_predictions(predictions, labels) 
        average_percentage_of_correct /= num_averaging_iterations
        
        self._reset_sample_state.run()
        
        if not isinstance(self._valid_size, dict): 
            validation_percentage_of_correct = 0.
            for _ in range(self._valid_size):
                b = self._valid_batches.next()
                predictions = self._sample_prediction.eval({self._sample_input: b[0]})
                validation_percentage_of_correct += percent_of_correct_predictions(predictions, b[1])
            validation_percentage_of_correct /= self._valid_size
            data_for_plot['validation']['percentage'].append(validation_percentage_of_correct)
        else:
            keys = self._valid_size.keys()
            validation_percentage_of_correct = dict([zipped for zipped in zip(keys, [0. for _ in keys])])
            for key in keys:
                for _ in range(self._valid_size[key]):
                    b = self._valid_batches[key].next()
                    predictions = self._sample_prediction.eval({self._sample_input: b[0]})
                    validation_percentage_of_correct[key] += percent_of_correct_predictions(predictions, b[1])
                validation_percentage_of_correct[key] /= self._valid_size[key]
                data_for_plot[key]['percentage'].append(validation_percentage_of_correct)
        
        
        data_for_plot['train']['percentage'].append(average_percentage_of_correct)
        
        return data_for_plot
        
    def simple_run(self,
                   num_averaging_iterations,
                   save_path,
                   min_num_steps,
                   loss_frequency,
                   block_of_steps,        #learning has a chance to be stopped after every block of steps
                   num_stairs,
                   decay,
                   stop_percent,
                   optional_feed_dict=None,
                   half_life_fixed=False,
                   fixed_num_steps=False):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        
        if not half_life_fixed and (self._last_num_steps > min_num_steps):
            min_num_steps = self._last_num_steps
        half_life = min_num_steps / num_stairs 
        
        with tf.Session(graph=self._graph, config=config) as session:
            learn_time = self.learn(session,
                                    min_num_steps,
                                    loss_frequency,
                                    block_of_steps,
                                    num_stairs,
                                    decay,
                                    stop_percent,
                                    optional_feed_dict=optional_feed_dict,
                                    half_life_fixed=half_life_fixed,
                                    fixed_num_steps=fixed_num_steps)
            data_for_plot = self.calculate_percentages(session, num_averaging_iterations)
            if self._last_num_steps == 0:
                GLOBAL_STEP = min_num_steps
            else:
                GLOBAL_STEP = self._last_num_steps
            lr = self._learning_rate.eval(feed_dict={self._global_step: GLOBAL_STEP,
                                                     self._half_life: half_life,
                                                     self._decay: decay},
                                                       session=session)  
            
            if save_path is not None:
                folder_list = save_path.split('/')[:-1]
                if len(folder_list) > 0:
                    current_folder = folder_list[0]
                    for idx, folder in enumerate(folder_list):
                        if idx > 0:
                            current_folder += ('/' + folder)
                        if not os.path.exists(current_folder):
                            os.mkdir(current_folder)


                self.saver.save(session, save_path)
        
        run_result = {"metadata": list(), "data": data_for_plot, "time": learn_time}
        if optional_feed_dict is None:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             num_averaging_iterations)
        else:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             num_averaging_iterations,
                                                             optional_feed_dict) 
        self._results.append(run_result)
        
        print("Number of steps = %s     Percentage = %.2f%%     Time = %.fs     Learning rate = %.4f" % 
              (run_result["metadata"][self._indices["num_steps"]],
              run_result["data"]["train"]["percentage"][-1],
              run_result["time"],
              lr))
        
        
    def run(self,
            num_stairs,
            decay,
            train_frequency,
            min_num_points,
            stop_percent,
            num_train_points_per_1_validation_point,
            averaging_number,
            optional_feed_dict=None,
            print_intermediate_results=False,
            half_life_fixed=False,                            # name of node which is to be printed
            add_operations=None, 
            add_text_operations=None, 
            print_steps=None,                       
            validation_add_operations=None,
            num_validation_prints=0,
            validation_example_length=None,
            fuse_texts=None):
        
        if not half_life_fixed and (self._last_num_steps > min_num_points * train_frequency):
            min_num_steps = self._last_num_steps
        else:
            min_num_steps = min_num_points * train_frequency
        half_life = min_num_steps / num_stairs
        
        if isinstance(self._valid_text, dict):
            data_for_plot = {'train': {'step': list(), 'percentage': list()}}
            for key in self._valid_text.keys():
                data_for_plot[key] = {'step': list(), 'percentage': list()}
        else:
            data_for_plot = {'train': {'step': list(), 'percentage': list()},
                             'validation': {'step': list(), 'percentage': list()}}
        
        if self._train_batches is None:
            self._train_batches = BatchGenerator(self._train_text,
                                                 self._batch_size,
                                                 self._vocabulary_size,
                                                 self._characters_positions_in_vocabulary,
                                                 self._num_unrollings)
        if self._valid_batches is None:
            if isinstance(self._valid_text, dict):
                keys = self._valid_text.keys()
                self._valid_batches = dict([zipped for zipped in zip(keys, [BatchGenerator(self._valid_text[key],
                                                                                           1,
                                                                                           self._vocabulary_size,
                                                                                           self._characters_positions_in_vocabulary,
                                                                                           1) for key in keys])])
            else:
                self._valid_batches = BatchGenerator(self._valid_text,
                                                     1,
                                                     self._vocabulary_size,
                                                     self._characters_positions_in_vocabulary,
                                                     1)
        
        averaging_number = min(averaging_number, train_frequency)
        averaging_step = max(train_frequency/averaging_number, 1)
        average_summing_started = False
        
        losses = [0.]

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(graph=self._graph, config=config) as session:
            tf.initialize_all_variables().run()
            if print_intermediate_results:
                print('Initialized')
            start_time = time.clock()
            average_percentage_of_correct = 0.
            mean_loss = 0
            step = 0
            validation_operations = [self._sample_prediction]

            if validation_add_operations is not None:
                real_to_print = list()
                for validation_add_operation in validation_add_operations:
                    exec('real_to_print.append(%s)' % validation_add_operation) 
            if validation_add_operations is not None:
                for real_operation in real_to_print:
                    if isinstance(real_operation, list):
                        if isinstance(real_operation[0], list):
                            for inner_list in real_operation:
                                validation_operations.extend(inner_list)
                        else:
                            validation_operations.extend(real_operation)
                    else:
                        validation_operations.append(real_operation)

            train_operations = [self._optimizer, self._loss, self._train_prediction, self._learning_rate]
            real_to_print_train = list()
            if add_text_operations is not None:
                for add_text_operation in add_text_operations:
                    exec('train_operations.append(%s)' % add_text_operation)
            if add_operations is not None:
                for train_operation in add_operations:
                    exec('real_to_print_train.append(%s)' % train_operation) 
            if add_operations is not None:
                for real_operation in real_to_print_train:
                    if isinstance(real_operation, list):
                        if isinstance(real_operation[0], list):
                            for inner_list in real_operation:
                                train_operations.extend(inner_list)
                        else:
                            train_operations.extend(real_operation)
                    else:
                        train_operations.append(real_operation)
            
            while not ((step - 1) % 10000 == 0 and
                       (losses[len(losses) / 2] - losses[-1]) < losses[len(losses) / 2] * stop_percent / 100 and
                       (len(losses) - 1) >= min_num_points / num_train_points_per_1_validation_point):
                batches = self._train_batches.next()
                feed_dict = {self._half_life: half_life, self._decay: decay}
                for i in range(self._num_unrollings + 1):
                    feed_dict[self._train_data[i]] = batches[i]
                if optional_feed_dict is not None:
                    new_dict = dict()
                    for optional_key in optional_feed_dict.keys():
                        exec("new_dict[%s] = %s" % (optional_key, optional_feed_dict[optional_key]))
                    feed_dict.update(new_dict)
                #print(train_operations)
                train_res = session.run(train_operations,
                                            feed_dict=feed_dict)
                l = train_res[1]
                predictions = train_res[2]
                lr = train_res[3]
                if add_text_operations is not None:
                    add_text_res = train_res[4:4+len(add_text_operations)]
                    if add_operations is not None:
                        add_train_res = train_res[4+len(add_text_operations):]
                else:
                    add_train_res = train_res[4:]
                #print('len(add_train_res) = %s' % len(add_train_res))
                if print_steps is not None:
                    if step in print_steps:        
                        print("step: %s" % step)
                        if add_text_operations is not None:
                            for print_text_op, add_text_op_res in zip(add_text_operations, add_text_res):
                                print('%s: ' % print_text_op)
                                add_text_chars = np.split(add_text_op_res, add_text_op_res.shape[0])
                                add_text = u''
                                for add_text_char in add_text_chars:
                                    add_text += characters(sample(add_text_char, self._vocabulary_size), self._vocabulary)[0] 
                                print(add_text)
                        if add_operations is not None:
                            print_counter = 0
                            for train_add_operation, real_operation in zip(add_operations, real_to_print_train):
                                if isinstance(real_operation, list):
                                    print('%s: ' % train_add_operation)
                                    if isinstance(real_operation[0], list):
                                        for list_idx in range(len(real_operation)):
                                            print(' '*2, '[%s]:' % list_idx)
                                            for tensor_idx in range(len(real_operation[0])):
                                                print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), add_train_res[print_counter])
                                                print_counter += 1
                                    else:
                                        for tensor_idx in range(len(real_operation)):
                                            print(' '*2, '[%s]:' % tensor_idx, add_train_res[print_counter])
                                            print_counter += 1
                                else:
                                    print('%s: ' % train_add_operation, add_train_res[print_counter])
                                    print_counter += 1
        
                mean_loss += l
                if ((step - (step / train_frequency) * train_frequency) % averaging_step == 0) and average_summing_started:
                    labels = np.concatenate(list(batches)[1:])
                    average_percentage_of_correct += percent_of_correct_predictions(predictions, labels)
                average_summing_started = True
                
                if step % train_frequency == 0:
                    if step > 0:
                        average_percentage_of_correct /= averaging_number
                        data_for_plot['train']['step'].append(step)
                        data_for_plot['train']['percentage'].append(average_percentage_of_correct)
                    if step % (train_frequency * num_train_points_per_1_validation_point) == 0:
                        if step > 0:
                            mean_loss = mean_loss / (train_frequency * num_train_points_per_1_validation_point)
                            # The mean loss is an estimate of the loss over the last few batches.

                            losses.append(mean_loss)
                        
                        if print_intermediate_results:
                            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                            print('Percentage_of correct: %.2f%%' % average_percentage_of_correct)
                            if step % (train_frequency * num_train_points_per_1_validation_point * 10) == 0:
                                # Generate some samples.
                                print("\nrandom:")
                                print('=' * 80)
                                for _ in range(5):
                                    feed = sample(random_distribution(self._vocabulary_size), self._vocabulary_size)
                                    sentence = characters(feed, self._vocabulary)[0]
                                    self._reset_sample_state.run()
                                    for _ in range(79):
                                        prediction = self._sample_prediction.eval({self._sample_input: feed})
                                        feed = sample(prediction, self._vocabulary_size)
                                        sentence += characters(feed, self._vocabulary)[0]
                                    print(sentence)
                                print('=' * 80)
                                if fuse_texts is not None:
                                    print("\nfrom fuse:")
                                    print('=' * 80)
                                    for fuse_idx, fuse in enumerate(fuse_texts):
                                        print("%s. fuse: %s" % (fuse_idx, fuse))
                                        fuse_list = list()
                                        for fuse_char in fuse:
                                            new_one_hot = np.zeros(shape=[1, self._vocabulary_size], dtype=np.float)
                                            new_one_hot[0, char2id(fuse_char, self._characters_positions_in_vocabulary)] = 1.
                                            fuse_list.append(new_one_hot)
                                        sentence = u''
                                        sentence += fuse
                                        self._reset_sample_state.run()
                                        for fuse_one_hot in fuse_list[:-1]:
                                            _ = self._sample_prediction.eval({self._sample_input: fuse_one_hot})
                                        feed = fuse_list[-1]
                                        for _ in range(79):
                                            prediction = self._sample_prediction.eval({self._sample_input: feed})
                                            feed = sample(prediction, self._vocabulary_size)
                                            sentence += characters(feed, self._vocabulary)[0]
                                        print(sentence)
                                    print('=' * 80)
                                
                        mean_loss = 0
                        # Measure validation set perplexity.
                        self._reset_sample_state.run()
                        validation_percentage_of_correct = 0.
                        if validation_example_length is not None:
                            fact = u''
                            predicted = u''
                        if isinstance(self._valid_batches, dict):
                            for key in self._valid_batches.keys():
                                for idx in range(self._valid_size[key]):
                                    b = self._valid_batches[key].next()
                                    validation_result = session.run(validation_operations,
                                                                    {self._sample_input: b[0]})
                                    validation_percentage_of_correct += percent_of_correct_predictions(validation_result[0], b[1])
                                    if print_intermediate_results:
                                        if validation_add_operations is not None:
                                            if idx < num_validation_prints:
                                                print('%s:' % idx)
                                                print_counter = 1
                                                for validation_add_operation, real_operation in zip(validation_add_operations, real_to_print):
                                                    if isinstance(real_operation, list):
                                                        print('%s: ' % validation_add_operation)
                                                        if isinstance(real_operation[0], list):
                                                            for list_idx in range(len(real_operation)):
                                                                print(' '*2, '[%s]:' % list_idx)
                                                                for tensor_idx in range(len(real_operation[0])):
                                                                    print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), validation_result[print_counter])
                                                                    print_counter += 1
                                                        else:
                                                            for tensor_idx in range(len(real_operation)):
                                                                print(' '*2, '[%s]:' % tensor_idx, validation_result[print_counter])
                                                                print_counter += 1
                                                    else:
                                                        print('%s: ' % validation_add_operation, validation_result[print_counter])
                                                        print_counter += 1
                                        if validation_example_length is not None:
                                            if idx < validation_example_length:
                                                fact += characters(sample(b[0], self._vocabulary_size), self._vocabulary)[0]
                                                predicted += characters(sample(validation_result[0], self._vocabulary_size), self._vocabulary)[0]
                                
                                validation_percentage_of_correct /= self._valid_size[key]
                                if print_intermediate_results:
                                    if validation_example_length is not None:
                                        print('%s example (input and output):' % key)
                                        print(fact)
                                        print(predicted)
                                    print('%s percentage of correct: %.2f%%\n' % (key, validation_percentage_of_correct))
                                data_for_plot[key]['step'].append(step)
                                data_for_plot[key]['percentage'].append(validation_percentage_of_correct)
                        else:
                            for idx in range(self._valid_size):
                                b = self._valid_batches.next()
                                validation_result = session.run(validation_operations,
                                                                {self._sample_input: b[0]})
                                validation_percentage_of_correct += percent_of_correct_predictions(validation_result[0], b[1])
                                if print_intermediate_results:
                                    if validation_add_operations is not None:
                                        if idx < num_validation_prints:
                                            print('%s:' % idx)
                                            print_counter = 1
                                            for validation_add_operation, real_operation in zip(validation_add_operations, real_to_print):
                                                if isinstance(real_operation, list):
                                                    print('%s: ' % validation_add_operation)
                                                    if isinstance(real_operation[0], list):
                                                        for list_idx in range(len(real_operation)):
                                                            print(' '*2, '[%s]:' % list_idx)
                                                            for tensor_idx in range(len(real_operation[0])):
                                                                print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), validation_result[print_counter])
                                                                print_counter += 1
                                                    else:
                                                        for tensor_idx in range(len(real_operation)):
                                                            print(' '*2, '[%s]:' % tensor_idx, validation_result[print_counter])
                                                            print_counter += 1
                                                else:
                                                    print('%s: ' % validation_add_operation, validation_result[print_counter])
                                                    print_counter += 1
                                    if validation_example_length is not None:
                                        if idx < validation_example_length:
                                            fact += characters(sample(b[0], self._vocabulary_size), self._vocabulary)[0]
                                            predicted += characters(sample(validation_result[0], self._vocabulary_size), self._vocabulary)[0]
                                
                            validation_percentage_of_correct /= self._valid_size
                            if print_intermediate_results:
                                if validation_example_length is not None:
                                    print('validation example (input and output):')
                                    print(fact)
                                    print(predicted)
                                print('Validation percentage of correct: %.2f%%\n' % validation_percentage_of_correct)
                            data_for_plot['validation']['step'].append(step)
                            data_for_plot['validation']['percentage'].append(validation_percentage_of_correct)
                    average_percentage_of_correct = 0.
                    average_summing_started = False
                step += 1
            finish_time = time.clock()
        self._last_num_steps = step
        run_result = {"metadata": list(), "data": data_for_plot, "time": (finish_time - start_time)}
        if optional_feed_dict is None:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             averaging_number)
        else:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             averaging_number,
                                                             optional_feed_dict) 
        self._results.append(run_result)

        
        if len(data_for_plot['train']['percentage']) < min_num_points:
            print("ERROR! failed to get enough data points")
        else:
            print("Number of steps = %s     Percentage = %.2f%%     Time = %.fs     Learning rate = %.4f" % 
                  (step,
                   sum(data_for_plot["train"]["percentage"][-1: -1 - min_num_points / 5: -1]) / (min_num_points / 5),
                   run_result["time"],
                   lr))

        
    def plot(self, results_numbers, plot_validation=False, save=False, save_folder=None):
        plt.close()
        colors = {0: 'k',
                  1: 'blue',
                  2: 'darkgoldenrod',
                  3: 'firebrick',
                  4: 'cyan',
                  5: 'gray',
                  6: 'm',
                  7: 'green',
                  8: 'yellow',
                  9: 'purple',
                  10: 'r',
                  11: '#E24A33',
                  12: '#92C6FF', 
                  13: '#0072B2',
                  14: '#30a2da',
                  15: '#4C72B0',
                  16: '#8EBA42',
                  17: '#6d904f'}
        fig = plt.figure(1)
        for i in range(len(results_numbers)):
            x_list = self._results[results_numbers[i]]["data"]['train']['step']
            y_list = self._results[results_numbers[i]]["data"]['train']['percentage']
            plt.plot(x_list, y_list, colors[i % 19])
            if plot_validation:
                if 'validation' in self._results[results_numbers[i]]["data"]:
                    x_list = self._results[results_numbers[i]]["data"]['validation']['step']
                    y_list = self._results[results_numbers[i]]["data"]['validation']['percentage']
                    plt.plot(x_list, y_list, color=colors[i], linestyle='dashed')
                else:
                    if len(results_numbers) > 1:
                        print('If several validations only 1 result is allowed')
                        return 0
                    validations_counter = 1
                    for key in self._results[results_numbers[i]]["data"]:
                        if key != 'train':
                            x_list = self._results[results_numbers[i]]["data"][key]['step']
                            y_list = self._results[results_numbers[i]]["data"][key]['percentage']
                            plt.plot(x_list, y_list, color=colors[validations_counter], linestyle='solid')
                            validations_counter += 1
        num_layers_string = self._results[results_numbers[i]]["metadata"][self._indices["num_layers"]]
        num_unrollings_string = self._results[results_numbers[i]]["metadata"][self._indices["num_unrollings"]]
        plt.title('Number of layers = %s; Number of unrollings = %s' % (num_layers_string,
                                                                        num_unrollings_string))
        

        plt.xlabel('step')
        plt.ylabel('percentage of correct')
        x1, x2, y1, y2 = plt.axis()
        
        text_labels = list()
        if 'validation' in self._results[results_numbers[0]]["data"]:
            for i in range(len(results_numbers)):
                text_label = ""
                text_label += ("time = %.fs;" % self._results[results_numbers[i]]["time"])
                text_label += (" half life = %s;" % self._results[results_numbers[i]]["metadata"][self._indices["half_life"]])
                text_label += (" decay = %.1f;" % self._results[results_numbers[i]]["metadata"][self._indices["decay"]])
                text_labels.append(text_label)
        else:
            for key in self._results[results_numbers[i]]["data"].keys():
                text_label = key
                text_labels.append(text_label)
        help_text = list()
        for i in range(len(text_labels)):
            vertical_position = y2 - float(y2 - y1) / (len(text_labels) + 1) * float(i+1)
            help_text.append(plt.text(x2 + 0.05 * (x2 - x1), vertical_position, text_labels[i],  va = 'center', ha = 'left', color=colors[i]))
        plt.grid()

        if save:
            nodes_string = "%s"
            if self._num_layers > 1:
                for i in range(self._num_layers - 1):
                    nodes_string += "_%s"
                nodes_string = '(' + nodes_string + ')'
            plot_filename = "nl%s_;nn_%s;nu_%s;bs_%s;emb_%s" % (self._num_layers,
                                                                nodes_string,
                                                                self._num_unrollings,
                                                                self._batch_size,
                                                                False)
            plot_filename = plot_filename % tuple(self._num_nodes)           
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            index = 0
            while os.path.exists(save_folder + '/' + plot_filename + '#' + str(index) + '.png'):
                index += 1
            plt.savefig(save_folder + '/' + plot_filename + '#' + str(index) + '.png')

        plt.show()


        text_right_edge = 0.
        for i in range(len(help_text)):
            if help_text[i].get_window_extent().get_points()[1][0] > text_right_edge:
                text_right_edge = help_text[i].get_window_extent().get_points()[1][0]   
        text_left_edge = help_text[0].get_window_extent().get_points()[0][0] 
        plt.close()
        
        """if save:
            fig = plt.figure(1)
            for i in range(len(results_numbers)):
                x_list = self._results[results_numbers[i]]["data"]['train']['step']
                y_list = self._results[results_numbers[i]]["data"]['train']['percentage']
                plt.plot(x_list, y_list, colors[i % 19])
                if plot_validation:
                    x_list = self._results[results_numbers[i]]["data"]['validation']['step']
                    y_list = self._results[results_numbers[i]]["data"]['validation']['percentage']
                    plt.plot(x_list, y_list, color=colors[i], linestyle='dashed')
            plt.xlabel('step')
            plt.ylabel('percentage of correct') 
            num_layers_string = self._results[results_numbers[i]]["metadata"][self._indices["num_layers"]]
            num_unrollings_string = self._results[results_numbers[i]]["metadata"][self._indices["num_unrollings"]]
            plt.title('Number of layers = %s; Number of unrollings = %s' % (num_layers_string,
                                                                            num_unrollings_string))
            for i in range(len(text_labels)):
                vertical_position = y2 - float(y2 - y1) / (len(text_labels) + 1) * float(i+1) 
                plt.text(x2 + 0.05 * (x2 - x1), vertical_position, text_labels[i],  va = 'center', ha = 'left', color=colors[i%18])
                
            coefficient = (2*text_left_edge - text_right_edge) / text_left_edge
            plt.tight_layout(rect=(0, 0, coefficient, 1))
            size = fig.get_size_inches()
            fig.set_size_inches(size[0]/coefficient, size[1])
            plt.grid()
            
            
            nodes_string = "%s"
            if self._num_layers > 1:
                for i in range(self._num_layers - 1):
                    nodes_string += "_%s"
                nodes_string = '(' + nodes_string + ')'
            plot_filename = "nl%s_;nn_%s;nu_%s;bs_%s;emb_%s" % (self._num_layers,
                                                                nodes_string,
                                                                self._num_unrollings,
                                                                self._batch_size,
                                                                False)
            plot_filename = plot_filename % tuple(self._num_nodes)           
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            index = 0
            while os.path.exists(save_folder + '/' + plot_filename + '#' + str(index) + '.png'):
                index += 1
            plt.savefig(save_folder + '/' + plot_filename + '#' + str(index) + '.png')
            plt.close()"""
        
        
    def destroy(self):
        tf.reset_default_graph()
        

    def run_for_analitics(self,
                          analitics_function,
                          save_path,
                          *args): # analitics arguments, learning arguments
        """learning arguments and analitic arguments are lists of arguments
        passed to learn and analitic_function methods"""
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        with tf.Session(graph=self._graph, config=config) as session:
            if save_path is None:
                learn_time = self.learn(session, *args[1])
            else:
                self.saver.restore(session, save_path)
            analitics_result = analitics_function(session, *args[0])
        return analitics_result

    def get_result(self,
                   restore_path,
                   num_averaging_iterations,
                   min_num_steps,
                   num_stairs,
                   decay,
                   optional_feed_dict=None,
                   fixed_num_steps=False):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        half_life = min_num_steps / num_stairs
        if self._train_batches is None:
            self._train_batches = BatchGenerator(self._train_text,
                                                 self._batch_size,
                                                 self._vocabulary_size,
                                                 self._characters_positions_in_vocabulary,
                                                 self._num_unrollings)

        with tf.Session(graph=self._graph, config=config) as session:
            self.saver.restore(session, restore_path)
            data_for_plot = self.calculate_percentages(session, num_averaging_iterations)
            GLOBAL_STEP = min_num_steps
            lr = self._learning_rate.eval(feed_dict={self._global_step: GLOBAL_STEP,
                                                     self._half_life: half_life,
                                                     self._decay: decay},
                                                       session=session)  

        
        run_result = {"metadata": list(), "data": data_for_plot, "time": 0.}
        if optional_feed_dict is None:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             num_averaging_iterations)
        else:
            run_result["metadata"] = self._generate_metadata(half_life,
                                                             decay,
                                                             num_averaging_iterations,
                                                             optional_feed_dict) 
        self._results.append(run_result)
        
        print("Number of steps = %s     Percentage = %.2f%%     Time = %.fs     Learning rate = %.4f" % 
              (run_result["metadata"][self._indices["num_steps"]],
              run_result["data"]["train"]["percentage"][-1],
              run_result["time"],
              lr))        
                   

    def get_insight(self, session, model_vars, intermediate_vars, length):
        self._reset_sample_state.run()
        self._valid_batches = BatchGenerator(self._valid_text,
                                             1,
                                             self._vocabulary_size,
                                             self._characters_positions_in_vocabulary,
                                             1)
        model_vars_list = list()
        for model_var in model_vars:
            exec('model_vars_list.append(%s)' % model_var)
        intermediate_vars_list = list()
        intermediate_results = dict()
        for intermediate_var in intermediate_vars:
            exec('intermediate_vars_list.append(%s)' % intermediate_var)
            intermediate_results[intermediate_var] = list()
        intermediate_vars_list.append(self._sample_prediction)
        model_results = dict()
        for idx, model_var in enumerate(model_vars):
            model_results[model_var] = model_vars_list[idx].eval()
        for _ in range(length):
            b = self._valid_batches.next()
            run_results = session.run(intermediate_vars_list, {self._sample_input: b[0]})
            for intermediate_var, run_result in zip(intermediate_vars, run_results[:-1]):
                intermediate_results[intermediate_var].append(run_result)
        return (model_results, intermediate_results, self._valid_text[:length])
