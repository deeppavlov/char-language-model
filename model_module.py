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
import matplotlib
import matplotlib.pyplot as plt
import codecs
import time
import os
import gc
from six.moves import cPickle as pickle
from tensorflow.python import debug as tf_debug

url = 'http://mattmahoney.net/dc/'

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
    for char in text:
        if ord(char) > 255:
            not_one_byte_counter += 1 
        if len(present_characters) <=  ord(char):
            present_characters.extend([0]*(ord(char) - len(present_characters) + 1))
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
    print(u'Unexpected character: %s\nUnexpected character number: %s\nUnexpected character has its place = %s\n' % (char, ord(char), present_characters_indices[i]))
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
  def __init__(self, text, batch_size, vocabulary_size, characters_positions_in_vocabulary,  num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._vocabulary_size = vocabulary_size
    self._characters_positions_in_vocabulary = characters_positions_in_vocabulary
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._start_batch()

  def _start_batch(self):
    batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id('\n', self._characters_positions_in_vocabulary)] = 1.0
    return batch    
  
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

def compute_perplexity(probabilities):
    probabilities[probabilities < 1e-10] = 1e-10
    log_probs = np.log2(probabilities)
    entropy_by_character = np.sum(- probabilities * log_probs, axis=1)
    return np.mean(np.exp2(entropy_by_character))

# bits per character
def compute_BPC(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    log_predictions = np.log2(predictions)
    BPC_by_character = np.sum(- labels * log_predictions, axis=1)
    return np.mean(BPC_by_character)

def compute_BPC_and_perplexity(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    log_predictions = np.log2(predictions)
    entropy_by_character = np.sum(- predictions * log_predictions, axis=1)
    perplexity_by_character = np.exp2(entropy_by_character)
    BPC_by_character = np.sum(- labels * log_predictions, axis=1)
    return np.mean(BPC_by_character), np.mean(perplexity_by_character)

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
              save_steps=None,                # steps at which model is saved into save_path + str(save_steps[i])
              save_path=None,                 # mask of paths which are used for saving model 
              optional_feed_dict=None,
              half_life_fixed=False,
              fixed_num_steps=False):
        if not half_life_fixed and (self._last_num_steps > min_num_steps):
            min_num_steps = self._last_num_steps
        half_life = min_num_steps // num_stairs
            
            
        if self._train_batches is None:
            self._train_batches = BatchGenerator(self._train_text,
                                                 self._batch_size,
                                                 self._vocabulary_size,
                                                 self._characters_positions_in_vocabulary,
                                                 self._num_unrollings)
            
        losses = [0.]

        session.run(tf.global_variables_initializer())
        start_time = time.clock()
        mean_loss = 0.
        step = 0
        if not fixed_num_steps:
            stop_condition = ((step - 1) % block_of_steps == 0 and
                                (losses[len(losses) // 2] - losses[-1]) < losses[len(losses) // 2] * stop_percent / 100 and
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
            if save_steps is not None:
                if step in save_steps:
                    current_save_path = save_path + str(step)
                    folder_list = current_save_path.split('/')[:-1]
                    if len(folder_list) > 0:
                        current_folder = folder_list[0]
                        for idx, folder in enumerate(folder_list):
                            if idx > 0:
                                current_folder += ('/' + folder)
                            if not os.path.exists(current_folder):
                                os.mkdir(current_folder)
                    self.saver.save(session, current_save_path)

            if not fixed_num_steps:
                stop_condition = ((step - 1) % block_of_steps == 0 and
                                    (losses[len(losses) // 2] - losses[-1]) < losses[len(losses) // 2] * stop_percent / 100 and
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
            data_for_plot = {'train': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()},
                             'validation': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}}
        else:
            keys = self._valid_text.keys()
            data_for_plot = {'train': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}} 
            for key in keys:
                data_for_plot[key] = {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}           
        
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
        average_BPC = 0
        average_perplexity = 0
        for _ in range(num_averaging_iterations):
            for _ in range(self.SKIP_LENGTH // self._num_unrollings):
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
            average_BPC += compute_BPC(predictions, labels)
            average_perplexity += compute_perplexity(predictions)
        average_percentage_of_correct /= num_averaging_iterations
        average_BPC /= num_averaging_iterations 
        average_perplexity /= num_averaging_iterations
        
        self._reset_sample_state.run()
        
        if not isinstance(self._valid_size, dict): 
            validation_percentage_of_correct = 0.
            validation_BPC = 0.
            validation_perplexity = 0.
            for _ in range(self._valid_size):
                b = self._valid_batches.next()
                predictions = self._sample_prediction.eval({self._sample_input: b[0]})
                validation_percentage_of_correct += percent_of_correct_predictions(predictions, b[1])
                current_BPC, current_perplexity = compute_BPC_and_perplexity(predictions, b[1])
                validation_BPC += current_BPC
                validation_perplexity += current_perplexity
            validation_percentage_of_correct /= self._valid_size
            validation_BPC /= self._valid_size
            validation_perplexity /= self._valid_size
            data_for_plot['validation']['percentage'].append(validation_percentage_of_correct)
            data_for_plot['validation']['BPC'].append(validation_BPC)
            data_for_plot['validation']['perplexity'].append(validation_perplexity)
        else:
            keys = self._valid_size.keys()
            validation_percentage_of_correct = dict([zipped for zipped in zip(keys, [0. for _ in keys])])
            validation_perplexity = dict([zipped for zipped in zip(keys, [0. for _ in keys])])
            validation_BPC = dict([zipped for zipped in zip(keys, [0. for _ in keys])])
            for key in keys:
                for _ in range(self._valid_size[key]):
                    b = self._valid_batches[key].next()
                    predictions = self._sample_prediction.eval({self._sample_input: b[0]})
                    validation_percentage_of_correct[key] += percent_of_correct_predictions(predictions, b[1])
                    current_BPC, current_perplexity = compute_BPC_and_perplexity(predictions, b[1])
                    validation_BPC[key] += current_BPC
                    validation_perplexity[key] += current_perplexity
                validation_percentage_of_correct[key] /= self._valid_size[key]
                validation_perplexity[key] /= self._valid_size[key]
                validation_BPC[key] /= self._valid_size[key]
                data_for_plot[key]['percentage'].append(validation_percentage_of_correct[key])
                data_for_plot[key]['perplexity'].append(validation_perplexity[key])
                data_for_plot[key]['BPC'].append(validation_BPC[key])
        
        
        data_for_plot['train']['percentage'].append(average_percentage_of_correct)
        data_for_plot['train']['perplexity'].append(average_perplexity)
        data_for_plot['train']['BPC'].append(average_BPC)
        
        return data_for_plot

    def split_to_path_name(self, path):
        parts = path.split('/')
        name = parts[-1]
        path = '/'.join(parts[:-1])
        return path, name 

    def create_path(self, path):
        folder_list = path.split('/')[:-1]
        if len(folder_list) > 0:
            current_folder = folder_list[0]
            for idx, folder in enumerate(folder_list):
                if idx > 0:
                    current_folder += ('/' + folder)
                if not os.path.exists(current_folder):
                    os.mkdir(current_folder)

    def loop_through_indices(self, filename, start_index):
        path, name = self.split_to_path_name(filename)
        if '.' in name:
            inter_list = name.split('.')
            extension = inter_list[-1]
            base = '.'.join(inter_list[:-1])
            base += '#%s' 
            name = '.'.join([base, extension])
            
        else:
            name += '#%s'
        if path != '':
            base_path = '/'.join([path, name])
        else:
            base_path = name
        index = start_index    
        while os.path.exists(base_path % index):
            index += 1
        return base_path % index

    def add_index_to_filename_if_needed(self, filename, index=None):
        if index is not None:
            return self.loop_through_indices(filename, index) 
        if os.path.exists(filename):
            return self.loop_through_indices(filename, 1)
        return filename
                       
        
    def save_graph(self, session, save_path):
        self.create_path(save_path)
        self.saver.save(session, save_path)

    def parse_collection_res(self, res, operation):
        if operation[1] == 'text':
            collection_text_res = res
            collection_operation_text_chars = np.split(collection_text_res, collection_text_res.shape[0])
            output = u''
            for collection_operation_text_char in collection_operation_text_chars:
                output += characters(sample(collection_operation_text_char, self._vocabulary_size), self._vocabulary)[0] 
        else:
            output = res
        return output


    def simple_run(self,
                   num_averaging_iterations,    # number of percents values used for final averaging
                   save_path,
                   min_num_steps,         # minimum number of learning iterations
                   loss_frequency,          # period of checking loss function. It is used defining if learning should be stopped
                   block_of_steps,        #learning has a chance to be stopped after every block of steps
                   num_stairs,             # number of times 'learning_rate' is multiplied on 'decay'
                   decay,                  # a factor by which the learning rate decreases each 'half_life'
                   stop_percent,              # if fixed_num_steps=False this parameter defines when the learning process should be stopped. If during half the total learning time loss function decreased less than by 'stop_percent' percents the learning would be stopped
                   save_steps=None,            # steps at which model is saved into save_path + str(save_steps[i])
                   optional_feed_dict=None,
                   half_life_fixed=False,
                   fixed_num_steps=False,
                   gpu_memory=None):
        config = tf.ConfigProto(allow_soft_placement=False,
                                log_device_placement=False)
        if gpu_memory is not None:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        
        if not half_life_fixed and (self._last_num_steps > min_num_steps):
            min_num_steps = self._last_num_steps
        half_life = min_num_steps // num_stairs
        if save_steps is None: 
            learn_save_path = None
        else:
            learn_save_path = save_path
        with tf.Session(graph=self._graph, config=config) as session:
            learn_time = self.learn(session,
                                    min_num_steps,
                                    loss_frequency,
                                    block_of_steps,
                                    num_stairs,
                                    decay,
                                    stop_percent,
                                    save_steps=save_steps,
                                    save_path=learn_save_path,
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
            
            self.save_graph(session, save_path)
        
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

    def pickle(self,
               path,
               name,
               dictionary):
        if not path == '':
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except Exception as e:
                    print("Unable create folder '%s'" % path, ':', e) 
        if path == '': 
            print('Pickling %s' % name)
        else:
            print('Pickling %s' % (path + '/' + name))
        try:
            if path == '': 
                with open(name, 'wb') as f:
                    pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(path + '/' + name, 'wb') as f:
                    pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', name, ':', e)             
        
        
    def run(self,
            num_stairs,                                       # number of times learning_rate is decreased while collecting min_num_points points for plot (half_life = min_num_points * train_frequency / num_stairs)
            decay,                                            # a factor by which learning_rate is decreased
            train_frequency,                                  # each 'train_frequency' steps loss and percent correctly predicted letters is calculated
            min_num_points,                                   # minimum number of times loss and percent correctly predicted letters are calculated while learning (train points)
            stop_percent,                                     # if during half total spent time loss decreased by less than 'stop_percent' percents learning process is stopped
            num_train_points_per_1_validation_point,          # when train point is obtained validation may be performed
            averaging_number,                                 # when train point percent is calculated results got on averaging_number chunks are averaged
            fixed_number_of_steps=None,                       # if not None 'fixed_number_of_steps' learning iterations will be performed
            optional_feed_dict=None,                          # it is sometimes needed to add variables to feed_dict in session.run 
            print_intermediate_results=False,                 # if True results on every train_frequency steps are printed
            half_life_fixed=False,                            # name of node which is to be printed
            add_operations=None,                              # list of names of python objects (could be tensors or lists of tensors or lists of lists of tensors) which values ought to be printed at steps 'print_steps'
            add_text_operations=None,                         # list of names of tensors which can be transformed into text and which should be printed while learning. Only first chunk in batch is printed
            print_steps=None,                                 # steps at which 'add_operations' and 'add_text_operations' are printed
            block_validation=False,                           # if True validation will not be performed
            validation_add_operations=None,                   # same as 'add_operations' only printed during processing of validation dataset
            num_validation_prints=0,                          # number of validation characters for which validation_add_operations are printed
            validation_example_length=None,                   # length of validation example chunk for which predictions are printed
            fuse_texts=None,                                  # list of strings which are used as beginnings of sentences. Agent tries to continue fuse_texts and prints following 79 characters on the screen
            debug=False,                                      # if True TensorFlow Debugger (tfdbg) Command-Line-Interface is enabled
            allow_soft_placement=False,
            log_device_placement=False,                       # passed to tf.ConfigProto
            save_path=None,                                   # path to file which will be used for saving graph. If path does not exist it will be created
            path_to_file_for_saving_prints=None,              # path to file where everything apointed for printing is saved
            collection_operations=None,                       # a list of tuples. Each tuple contains two elements. First is operation name. Second indicates if operation encodes text or not ('text', 'number'). If 'text' before adding to collection numpy array will be transformed to str
            collection_steps=None,
            path_to_file_for_saving_collection=None,          # all add operations results will be added to a dictionary which will be pickled
            summarizing_logdir=None,                          # 'summarizing_logdir' is a path to directory for storing summaries
            summary_dict=None,                                # a dictionary containing 'summaries_collection_frequency', 'summary_tensors'
                                                              # every 'summaries_collection_frequency'-th step summary is collected 
                                                              # 'summary_tensors' is a list of collected summary tensors
            summary_graph=False,                              # defines if graph should be added to the summary                        
            gpu_memory=None):
                                                              
  
        BPC_coef = 1./np.log(2)

        if path_to_file_for_saving_prints is not None:
            self.create_path(path_to_file_for_saving_prints)
            file_object = open(path_to_file_for_saving_prints, 'w')
      
        def choose_where_to_print(*inputs):
            if print_intermediate_results:
                print(*inputs)
            if path_to_file_for_saving_prints is not None:
                for inp in inputs:
                    file_object.write(str(inp))
                file_object.write('\n')
                
        if not half_life_fixed and (self._last_num_steps > min_num_points * train_frequency):
            min_num_steps = self._last_num_steps
        else:
            min_num_steps = min_num_points * train_frequency
        half_life = min_num_steps // num_stairs
        
        if isinstance(self._valid_text, dict):
            data_for_plot = {'train': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}}
            for key in self._valid_text.keys():
                data_for_plot[key] = {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}
        else:
            data_for_plot = {'train': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()},
                             'validation': {'step': list(), 'percentage': list(), 'BPC': list(), 'perplexity': list()}}
        
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
        averaging_step = max(train_frequency//averaging_number, 1)
        average_summing_started = False
        
        losses = [0.]

        config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                log_device_placement=log_device_placement)
        if gpu_memory is not None:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        with tf.Session(graph=self._graph, config=config) as session:
            if debug:
                session = tf_debug.LocalCLIDebugWrapperSession(session)
                session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            session.run(tf.global_variables_initializer())
            choose_where_to_print('Initialized')
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
                    if isinstance(real_operation, list) or isinstance(real_operation, tuple):
                        if isinstance(real_operation[0], list) or isinstance(real_operation[0], tuple):
                            for inner_list in real_operation:
                                validation_operations.extend(inner_list)
                        else:
                            validation_operations.extend(real_operation)
                    else:
                        validation_operations.append(real_operation)

            # composing the list of operations which should be performed while training
            # 'train_operations' is a list which will be actually passed to 'session.run' func
            train_operations = [self._optimizer, self._loss, self._train_prediction, self._learning_rate]
            train_operations_map = list()
            # 'train_operations_map' is a list of length 6 (2 for edges of 'add_text_operations', 'add_operations', 'summary_dict['summary_tensors']')
            # 'real_to_print_train' is a list of python objects constructed from 'add_operations'. It is done because 'add_operations' may contain lists which should be preprocessed.
            real_to_print_train = list()
            # since 'add_text_operations' contains tensors, its contents added to 'train_operations' directly
            pointer = 4             # pointer is a variable used for calculating borders of 'add_text_operations', 'add_operations', 'summary_dict['summary_tensors']' in 'train_operations'
            if add_text_operations is not None:
                train_operations_map.append(pointer)
                pointer += len(add_text_operations)
                train_operations_map.append(pointer)
                for add_text_operation in add_text_operations:
                    exec('train_operations.append(%s)' % add_text_operation)
            else:
                train_operations_map.append(None)
                train_operations_map.append(None)
            # 'add_operations' added to 'real_to_print_train'
            if add_operations is not None:
                for train_operation in add_operations:
                    exec('real_to_print_train.append(%s)' % train_operation) 
            # 'real_to_print_train' is processed
            if add_operations is not None:
                train_operations_map.append(pointer)
                for real_operation in real_to_print_train:
                    if isinstance(real_operation, list) or isinstance(real_operation, tuple):
                        if isinstance(real_operation[0], list) or isinstance(real_operation[0], tuple):
                            for inner_list in real_operation:
                                pointer += len(inner_list)
                                train_operations.extend(inner_list)
                        else:
                            pointer += len(real_operation)
                            train_operations.extend(real_operation)
                    else:
                        pointer += 1
                        train_operations.append(real_operation)
                train_operations_map.append(pointer)
            else:
                train_operations_map.append(None)
                train_operations_map.append(None)

            if summarizing_logdir is not None:
                writer = tf.summary.FileWriter(summarizing_logdir)
                if summary_graph:
                     writer.add_graph(self._graph) 
                     # 'summary_dict['summary_tensors']' is processed
            if summary_dict is not None:
                train_operations_map.append(pointer)
                pointer += len(summary_dict['summary_tensors'])
                train_operations_map.append(pointer)
                #print("summary_dict['summary_tensors']")
                for tensor_name in summary_dict['summary_tensors']:
                    #print(tensor_name)
                    exec('train_operations.append(%s)' % tensor_name)
                #print("summary_dict['summary_tensors'] finished")
            else:
                train_operations_map.append(None)
                train_operations_map.append(None)

            real_to_print_collection = list()
            if collection_operations is not None:
                collection_dictionary = {'step': list()}
                for collection_operation in collection_operations:
                    collection_dictionary[collection_operation[0]] = list()
                    exec('real_to_print_collection.append(%s)' % collection_operation[0])
            else:
                train_operations_map.append(None)
                train_operations_map.append(None)
            if collection_operations is not None:
                train_operations_map.append(pointer)
                for real_collection_operation in real_to_print_collection:
                    if isinstance(real_collection_operation, list) or isinstance(real_collection_operation, tuple):
                        if isinstance(real_collection_operation[0], list) or isinstance(real_collection_operation[0], tuple):
                            for inner_list in real_collection_operation:
                                pointer += len(inner_list)
                                train_operations.extend(inner_list)
                        else:
                            pointer += len(real_collection_operation)
                            train_operations.extend(real_collection_operation)
                    else:
                        pointer += 1
                        train_operations.append(real_collection_operation)
                train_operations_map.append(pointer)

            def fixed_logical_factor(step_parameter):
                # returns True if learning should continue
                if fixed_number_of_steps is None:
                    return True
                else:
                    return step_parameter != fixed_number_of_steps
            
            # if 'fixed_number_of_steps' is None system stops learning when all of the following conditions are fulfilled
            # 1. (step - 1) % 10000 == 0
            # 2. difference between loss on step ('step'//2) and loss on current step is less the 'stop_percent' percents of loss on step ('step'//2)
            # 3. at least 'min_num_points' elements are in 'losses' list
            # if 'fixed_number_of_steps' is not None learning stops when  step == fixed_number_of_steps
            while (not ((step - 1) % 10000 == 0 and
                        (losses[len(losses) // 2] - losses[-1]) < losses[len(losses) // 2] * stop_percent / 100 and
                        (len(losses) - 1) >= min_num_points // num_train_points_per_1_validation_point)) and fixed_logical_factor(step):
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
                    add_text_res = train_res[train_operations_map[0]:train_operations_map[1]]
                if add_operations is not None:
                    add_train_res = train_res[train_operations_map[2]:train_operations_map[3]]
                if summary_dict is not None:
                    summary_res = train_res[train_operations_map[4]:train_operations_map[5]]
                if collection_operations is not None:
                    collection_res = train_res[train_operations_map[6]:train_operations_map[7]]
                if collection_operations is not None:
                    if step in collection_steps:
                        print_counter = 0
                        collection_dictionary['step'].append(step)
                        for collection_operation, real_operation in zip(collection_operations, real_to_print_collection):
                            if isinstance(real_operation, list) or isinstance(real_operation, tuple):
                                choose_where_to_print('%s: ' % train_add_operation)
                                if isinstance(real_operation[0], list) or isinstance(real_operation[0], tuple):
                                    storage = list()
                                    for list_idx in range(len(real_operation)):
                                        storage_item = list()
                                        for tensor_idx in range(len(real_operation[list_idx])):
                                            storage_item.append(self.parse_collection_res(collection_res[print_counter], collection_operation))
                                            print_counter += 1
                                        storage.append(storage_item)
                                    collection_dictionary[collection_operation[0]].append(storage)
                                else:
                                    storage = list()
                                    for tensor_idx in range(len(real_operation)):
                                        storage_item.append(self.parse_collection_res(collection_res[print_counter], collection_operation))
                                        print_counter += 1
                                    collection_dictionary[collection_operation[0]].append(storage)
                            else:
                                collection_dictionary[collection_operation[0]].append(self.parse_collection_res(collection_res[print_counter], collection_operation))
                                print_counter += 1
                    
                # print('len(add_train_res) = %s' % len(add_train_res))
                # implementing printing of 'add_text_operations' and 'add_operations'
                if print_steps is not None:
                    if step in print_steps:        
                        print("step: %s" % step)
                        if add_text_operations is not None:
                            for print_text_op, add_text_op_res in zip(add_text_operations, add_text_res):
                                choose_where_to_print('%s: ' % print_text_op)
                                add_text_chars = np.split(add_text_op_res, add_text_op_res.shape[0])
                                add_text = u''
                                for add_text_char in add_text_chars:
                                    add_text += characters(sample(add_text_char, self._vocabulary_size), self._vocabulary)[0] 
                                choose_where_to_print(add_text)
                        if add_operations is not None:
                            print_counter = 0
                            for train_add_operation, real_operation in zip(add_operations, real_to_print_train):
                                if isinstance(real_operation, list) or isinstance(real_operation, tuple):
                                    choose_where_to_print('%s: ' % train_add_operation)
                                    if isinstance(real_operation[0], list) or isinstance(real_operation[0], tuple):
                                        for list_idx in range(len(real_operation)):
                                            choose_where_to_print(' '*2, '[%s]:' % list_idx)
                                            for tensor_idx in range(len(real_operation[list_idx])):
                                                choose_where_to_print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), add_train_res[print_counter])
                                                print_counter += 1
                                    else:
                                        for tensor_idx in range(len(real_operation)):
                                            choose_where_to_print(' '*2, '[%s]:' % tensor_idx, add_train_res[print_counter])
                                            print_counter += 1
                                else:
                                    choose_where_to_print('%s: ' % train_add_operation, add_train_res[print_counter])
                                    print_counter += 1

                # adding summaries
                if summary_dict is not None: 
                    if step % summary_dict['summaries_collection_frequency'] == 0:
                        for res in summary_res:
                            writer.add_summary(res, step)
        
                mean_loss += l
                if ((step - (step // train_frequency) * train_frequency + 1) % averaging_step == 0) and average_summing_started:
                    labels = np.concatenate(list(batches)[1:])
                    average_percentage_of_correct += percent_of_correct_predictions(predictions, labels)
                average_summing_started = True

                                    
                if step % train_frequency == 0:
                    if step > 0:
                        average_percentage_of_correct /= averaging_number
                        data_for_plot['train']['step'].append(step)
                        data_for_plot['train']['percentage'].append(average_percentage_of_correct)
                        data_for_plot['train']['BPC'].append(BPC_coef * l)
                        data_for_plot['train']['perplexity'].append(compute_perplexity(predictions))
                    if not block_validation:
                        if step % (train_frequency * num_train_points_per_1_validation_point) == 0:
                            if step > 0:
                                mean_loss = mean_loss / (train_frequency * num_train_points_per_1_validation_point)
                                # The mean loss is an estimate of the loss over the last few batches.

                                losses.append(mean_loss)
                            
                            if print_intermediate_results or (path_to_file_for_saving_prints is not None):
                                choose_where_to_print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                                choose_where_to_print('Percentage_of correct: %.2f%%' % average_percentage_of_correct)
                                if step % (train_frequency * num_train_points_per_1_validation_point * 10) == 0:
                                    # Generate some samples.
                                    choose_where_to_print("\nrandom:")
                                    choose_where_to_print('=' * 80)
                                    for _ in range(5):
                                        feed = sample(random_distribution(self._vocabulary_size), self._vocabulary_size)
                                        sentence = characters(feed, self._vocabulary)[0]
                                        self._reset_sample_state.run()
                                        for _ in range(79):
                                            prediction = self._sample_prediction.eval({self._sample_input: feed})
                                            feed = sample(prediction, self._vocabulary_size)
                                            sentence += characters(feed, self._vocabulary)[0]
                                        choose_where_to_print(sentence)
                                    choose_where_to_print('=' * 80)
                                    if fuse_texts is not None:
                                        choose_where_to_print("\nfrom fuse:")
                                        choose_where_to_print('=' * 80)
                                        for fuse_idx, fuse in enumerate(fuse_texts):
                                            choose_where_to_print("%s. fuse: %s" % (fuse_idx, fuse))
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
                                            choose_where_to_print(sentence)
                                        choose_where_to_print('=' * 80)
                                    
                            mean_loss = 0
                            # Measure validation set perplexity.
                            self._reset_sample_state.run()
                            if validation_example_length is not None:
                                fact = u''
                                predicted = u''
                            if isinstance(self._valid_batches, dict):
                                for key in self._valid_batches.keys():
                                    validation_percentage_of_correct = 0.
                                    validation_BPC = 0.
                                    validation_perplexity = 0.
                                    for idx in range(self._valid_size[key]):
                                        b = self._valid_batches[key].next()
                                        validation_result = session.run(validation_operations,
                                                                        {self._sample_input: b[0]})
                                        validation_percentage_of_correct += percent_of_correct_predictions(validation_result[0], b[1])
                                        cur_BPC, cur_perplex = compute_BPC_and_perplexity(validation_result[0], b[1])
                                        validation_BPC += cur_BPC
                                        validation_perplexity += cur_perplex
                                        if print_intermediate_results or (path_to_file_for_saving_prints is not None):
                                            if validation_add_operations is not None:
                                                if idx < num_validation_prints:
                                                    choose_where_to_print('%s:' % idx)
                                                    print_counter = 1
                                                    for validation_add_operation, real_operation in zip(validation_add_operations, real_to_print):
                                                        if isinstance(real_operation, list):
                                                            choose_where_to_print('%s: ' % validation_add_operation)
                                                            if isinstance(real_operation[0], list):
                                                                for list_idx in range(len(real_operation)):
                                                                    choose_where_to_print(' '*2, '[%s]:' % list_idx)
                                                                    for tensor_idx in range(len(real_operation[0])):
                                                                        choose_where_to_print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), validation_result[print_counter])
                                                                        print_counter += 1
                                                            else:
                                                                for tensor_idx in range(len(real_operation)):
                                                                    choose_where_to_print(' '*2, '[%s]:' % tensor_idx, validation_result[print_counter])
                                                                    print_counter += 1
                                                        else:
                                                            choose_where_to_print('%s: ' % validation_add_operation, validation_result[print_counter])
                                                            print_counter += 1
                                            if validation_example_length is not None:
                                                if idx < validation_example_length:
                                                    fact += characters(sample(b[0], self._vocabulary_size), self._vocabulary)[0]
                                                    predicted += characters(sample(validation_result[0], self._vocabulary_size), self._vocabulary)[0]
                                    
                                    validation_percentage_of_correct /= self._valid_size[key]
                                    validation_BPC /= self._valid_size[key]
                                    validation_perplexity /= self._valid_size[key]
                                    if print_intermediate_results or (path_to_file_for_saving_prints is not None):
                                        if validation_example_length is not None:
                                            choose_where_to_print('%s example (input and output):' % key)
                                            choose_where_to_print('input:')
                                            choose_where_to_print(fact)
                                            choose_where_to_print('********************\noutput:')
                                            choose_where_to_print(predicted)
                                            choose_where_to_print('********************')
                                        choose_where_to_print('%s percentage of correct: %.2f%%\n' % (key, validation_percentage_of_correct))
                                    data_for_plot[key]['step'].append(step)
                                    data_for_plot[key]['percentage'].append(validation_percentage_of_correct)
                                    data_for_plot[key]['BPC'].append(validation_BPC)
                                    data_for_plot[key]['perplexity'].append(validation_perplexity)
                            else:
                                validation_percentage_of_correct = 0.
                                validation_BPC = 0.
                                validation_perplexity = 0.
                                for idx in range(self._valid_size):
                                    b = self._valid_batches.next()
                                    validation_result = session.run(validation_operations,
                                                                    {self._sample_input: b[0]})
                                    #print('validation_result[0] =', validation_result[0], '\nb[1] =', b[1]) 
                                    validation_percentage_of_correct += percent_of_correct_predictions(validation_result[0], b[1])
                                    #cur_BPC, cur_perplex = compute_BPC_and_perplexity(validation_result[0], b[1])
                                    cur_BPC, _ = compute_BPC_and_perplexity(validation_result[0], b[1])
                                    cur_perplex = compute_perplexity(validation_result[0])
                                    validation_BPC += cur_BPC
                                    validation_perplexity += cur_perplex
                                    if print_intermediate_results or (path_to_file_for_saving_prints is not None):
                                        if validation_add_operations is not None:
                                            if idx < num_validation_prints:
                                                choose_where_to_print('%s:' % idx)
                                                print_counter = 1
                                                for validation_add_operation, real_operation in zip(validation_add_operations, real_to_print):
                                                    if isinstance(real_operation, list) or isinstance(real_operation, tuple):
                                                        choose_where_to_print('%s: ' % validation_add_operation)
                                                        if isinstance(real_operation[0], list) or isinstance(real_operation[0], tuple):
                                                            for list_idx in range(len(real_operation)):
                                                                choose_where_to_print(' '*2, '[%s]:' % list_idx)
                                                                for tensor_idx in range(len(real_operation[0])):
                                                                    choose_where_to_print(' '*4, '[%s][%s]:' % (list_idx, tensor_idx), validation_result[print_counter])
                                                                    print_counter += 1
                                                        else:
                                                            for tensor_idx in range(len(real_operation)):
                                                                choose_where_to_print(' '*2, '[%s]:' % tensor_idx, validation_result[print_counter])
                                                                print_counter += 1
                                                    else:
                                                        choose_where_to_print('%s: ' % validation_add_operation, validation_result[print_counter])
                                                        print_counter += 1
                                        if validation_example_length is not None:
                                            if idx < validation_example_length:
                                                fact += characters(sample(b[0], self._vocabulary_size), self._vocabulary)[0]
                                                predicted += characters(sample(validation_result[0], self._vocabulary_size), self._vocabulary)[0]
                                    
                                validation_percentage_of_correct /= self._valid_size
                                validation_BPC /= self._valid_size
                                validation_perplexity /= self._valid_size
                                if print_intermediate_results or (path_to_file_for_saving_prints is not None):
                                    if validation_example_length is not None:
                                        choose_where_to_print('validation example (input and output):')
                                        choose_where_to_print('input:')
                                        choose_where_to_print(fact)
                                        choose_where_to_print('********************\noutput:')
                                        choose_where_to_print(predicted)
                                        choose_where_to_print('********************')
                                    choose_where_to_print('Validation percentage of correct: %.2f%%\n' % validation_percentage_of_correct)
                                data_for_plot['validation']['step'].append(step)
                                data_for_plot['validation']['percentage'].append(validation_percentage_of_correct)
                                data_for_plot['validation']['BPC'].append(validation_BPC)
                                data_for_plot['validation']['perplexity'].append(validation_perplexity)
                    average_percentage_of_correct = 0.
                    average_summing_started = False
                step += 1
            finish_time = time.clock()
            if save_path is not None:
                self.save_graph(session, save_path)
        if path_to_file_for_saving_collection is not None:
            path_to_folder, pickle_name = self.split_to_path_name(path_to_file_for_saving_collection)
            self.pickle(path_to_folder, pickle_name, collection_dictionary)
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

        
        choose_where_to_print("Number of steps = %s     Percentage = %.2f%%     Time = %.fs     Learning rate = %.4f" % 
              (step,
               sum(data_for_plot["train"]["percentage"][-1: -1 - min_num_points // 5: -1]) / (min_num_points // 5),
               run_result["time"],
               lr))

    def model_parameters_dump(self):
        dump = ""
        dump += "\n\nMODEL PARAMETERS\n" 
        dump += "number of layers: %s\n" % self._num_layers
        num_nodes_template = "number of nodes: ["
        for _ in range(self._num_layers):
            num_nodes_template += '%s ,'
        num_nodes_template = num_nodes_template[:-2] + ']\n'
        dump += num_nodes_template % tuple(self._num_nodes)
        dump += "number of unrollings: %s\n" % self._num_unrollings
        dump += "number of batch size: %s\n" % self._batch_size
        dump += "embedding size: %s\n" % self._embedding_size
        dump += "output embedding size: %s\n" % self._output_embedding_size
        dump += "init_parameter: %s\n" % self._init_parameter
        dump += "matr_init_parameter: %s\n" % self._matr_init_parameter
        dump += '\n'
        return dump


    def inference(self, log_path, checkpoint_path=None, gpu_memory=None, appending=True, add_model_parameters=True):
        self.create_path(log_path)
        if not appending:
            log_path = self.add_index_to_filename_if_needed(log_path)
        if appending:
            file_object = open(log_path, 'a')
        else:
            file_object = open(log_path, 'w')

        def print_and_log(*inputs, log=True, _print=True):
            if _print:
                print(*inputs)
            if log:
                for inp in inputs:
                    file_object.write(str(inp))
                file_object.write('\n')  
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        if gpu_memory is not None:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory


        file_object.write('\n*********************')
        if add_model_parameters:
            file_object.write(self.model_parameters_dump())

        with tf.Session(graph=self._graph, config=config) as session:
            if checkpoint_path is None:
                print_and_log('Skipping variables restoring. Continueing on current variables values')
            else:
                print('restoring from %s' % checkpoint_path)
                self.saver.restore(session, checkpoint_path)
            self._reset_sample_state.run()
            human_replica = input('Human: ')
            while not human_replica == 'FINISH':
                print_and_log('Human: '+human_replica, _print=False)
                for char in human_replica:
                    feed = char2vec(char, self._characters_positions_in_vocabulary) 
                    _ = self._sample_prediction.eval({self._sample_input: feed}) 

                # language generation
                feed = char2vec('\n', self._characters_positions_in_vocabulary)  
                prediction = self._sample_prediction.eval({self._sample_input: feed})
                counter = 0
                char = None
                bot_replica = ""
                while char != '\n' or counter > 500: 
                    feed = sample(prediction, self._vocabulary_size)
                    prediction = self._sample_prediction.eval({self._sample_input: feed})
                    char = characters(feed, self._vocabulary)[0]   
                    if char != '\n':
                        bot_replica += char
                    counter += 1
                print_and_log('Bot: '+bot_replica)
                feed = char2vec('\n', self._characters_positions_in_vocabulary)  
                _ = self._sample_prediction.eval({self._sample_input: feed}) 

                human_replica = input('Human: ')

        file_object.write('\n*********************')
        file_object.close()
                                     
                      

    def plot(self, xlists, ylists, labels, y_label, title=None, save_folder=None, show=False):
        plt.close()
        fig = plt.figure(1)
        for line_idx, ylist in enumerate(ylists):
            plt.plot(xlists[line_idx], ylist, colors[line_idx % 19])
        plt.xlabel('step')
        plt.ylabel(y_label)        
        x1, x2, y1, y2 = plt.axis()
        help_text = list()
        for label_idx, label in enumerate(labels):
            vertical_position = y2 - float(y2 - y1) / (len(labels) + 1) * float(label_idx+1)
            help_text.append(plt.text(x2 + 0.05 * (x2 - x1), vertical_position, label,  va = 'center', ha = 'left', color=colors[label_idx]))
        plt.grid()
        num_nodes_list = ['%s' for _ in range(self._num_layers)]
        num_nodes_string = '[' + ", ".join(num_nodes_list) + ']'
        num_nodes_string = num_nodes_string % tuple(self._num_nodes)
        if title is None:
            plt.title('Number of nodes = %s; Number of unrollings = %s' % (num_nodes_string,
                                                                           self._num_unrollings))
        else:
            plt.title(title)
        if save_folder is not None:
            r = fig.canvas.get_renderer()
            if len(help_text) != 0:
                fig.set_size_inches(8, 5)
                max_right = help_text[0].get_window_extent(renderer = r).get_points()[1][0]
                min_left = plt.axes().get_window_extent().get_points()[1][0]
                for text in help_text[1:]:
                    x1 = text.get_window_extent(renderer = r).get_points()[1][0]
                    if x1 > max_right:
                        max_right = x1
                coef = (2*min_left - max_right) / min_left
                plt.tight_layout(rect=(0, 0, coef, 1))
            else:
                print("There is no labels on plot")
            nodes_string = "%s"
            if self._num_layers > 1:
                for i in range(self._num_layers - 1):
                    nodes_string += "_%s"
                nodes_string = '(' + nodes_string + ')'
            plot_filename = y_label + '_' + "nl%s_;nn_%s;nu_%s;bs_%s;emb_%s" % (self._num_layers,
                                                                                nodes_string,
                                                                                self._num_unrollings,
                                                                                self._batch_size,
                                                                                hasattr(self, '_embedding_size'))
            plot_filename = plot_filename % tuple(self._num_nodes)           
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            index = 0
            while os.path.exists(save_folder + '/' + plot_filename + '#' + str(index) + '.png'):
                index += 1
            plt.savefig(save_folder + '/' + plot_filename + '#' + str(index) + '.png')
        if show:
            plt.show()

        

    def plot_all(self, results_numbers, plot_validation=False, indent=0, save_folder=None, show=False):
        # result_numbers is a list of numbers of runs to be plotted
        # indent is number of point from which the line is starting. It is convinient because first points can be definite outliers
        percentage_ylists = list()
        perplexity_ylists = list()
        BPC_ylists = list()
        xlists = list()
        labels = list()
        def add_key_all(result_num, Key):
            percentage_ylists.append(self._results[result_num]["data"][Key]['percentage'][indent:])
            labels.append(Key+'#%s'%result_num)   
            perplexity_ylists.append(self._results[result_num]["data"][Key]['perplexity'][indent:])
            BPC_ylists.append(self._results[result_num]["data"][Key]['BPC'][indent:])
            xlists.append(self._results[result_num]["data"][Key]['step'][indent:])

        for result_number in results_numbers:
            add_key_all(result_number, 'train')
            if plot_validation: 
                for key in self._results[result_number]["data"].keys():
                    if key != 'train':
                        add_key_all(result_number, key)
             
        """if len(results_numbers) > 1:
            for result_number in results_numbers:
                add_key_all(result_number, 'train')
                if plot_validation: 
                    for key in self._results[result_number]["data"].keys():
                        if key != 'train':
                            add_key_all(result_number, key)
        else:
            result_number = results_numbers[0]
            add_key_all(result_number, 'train')
            for key in self._results[result_number]["data"].keys():
                if plot_validation:
                    if key != 'train':
                        add_key_all(result_number, key)"""

        self.plot(xlists, percentage_ylists, labels, 'percentage', save_folder=save_folder, show=show)
        self.plot(xlists, perplexity_ylists, labels, 'perplexity', save_folder=save_folder, show=show)
        self.plot(xlists, BPC_ylists, labels, 'BPC', save_folder=save_folder, show=show)

    def plot_all_different(self, result_numbers, names, dataset, indent=0, save_folder=None, show=False):
        # result_numbers is a list of numbers of runs to be plotted
        # indent is number of point from which the line is starting. It is convinient because first points can be definite outliers
        # 'names' is a list of model names
        # 'dataset' is a string describing dataset used for plotting 'train', 'validation' or another
        percentage_ylists = list()
        perplexity_ylists = list()
        BPC_ylists = list()
        xlists = list()
        labels = list()
        def add_name(result_num, name):
            percentage_ylists.append(self._results[result_num]["data"][dataset]['percentage'][indent:])
            labels.append(name)   
            perplexity_ylists.append(self._results[result_num]["data"][dataset]['perplexity'][indent:])
            BPC_ylists.append(self._results[result_num]["data"][dataset]['BPC'][indent:])
            xlists.append(self._results[result_num]["data"][dataset]['step'][indent:])         
        for result_num, name in zip(result_numbers, names):
            add_name(result_num, name)
        self.plot(xlists, percentage_ylists, labels, 'percentage', save_folder=save_folder, show=show)
        self.plot(xlists, perplexity_ylists, labels, 'perplexity', save_folder=save_folder, show=show)
        self.plot(xlists, BPC_ylists, labels, 'BPC', save_folder=save_folder, show=show)
        
    def destroy(self):
        tf.reset_default_graph()
        

    def run_for_analitics(self,
                          analitics_function,
                          save_path,
                          *args): # analitics arguments, learning arguments
        """learning arguments and analitic arguments are lists of arguments
        passed to learn and analitic_function methods"""
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

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
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        half_life = min_num_steps // num_stairs
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


              
