import os
import pickle
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from some_useful_functions import (construct, add_index_to_filename_if_needed, match_two_dicts, create_path,
                                   paste_into_nested_structure, check_if_key_in_nested_dict,
                                   search_in_nested_dictionary, add_missing_to_list, print_and_log,
                                   apply_temperature, sample)


class Controller(object):
    """Controller is a class which instances are used for computing changing learning parameters. For example
    learning rate. It is also responsible for training stopping
    Usage:
        1. Construct controller by passing him 'storage' (a dictionary with monitored parameters, usually if used in
        _train method of Environment class 'storage' is self._storage) and specifications. Specifications is
        a dictionary with necessary parameters for computing the controlled value
        2. Use 'get' method to get current value
        3. If you wish to use Controller instance with new specifications you should:
            - add new private method to Controller which will be responsible for processing new specifications. It
              should take no arguments and return monitored value.
            - add elif entry in __init__ method for assigning 'get' with newly created method
            - if new approach requires parameters not provided in self._storage than add them. Also don't forget
              to pass this parameters to _update_storage method in the bottom of _train"""
    @staticmethod
    def create_change_tracking_specifications(specifications):
        if isinstance(specifications, list):
            old_specs = construct(specifications)
        if isinstance(specifications, dict):
            old_specs = [dict(specifications)]
        new_specs = dict()
        new_specs['old_specs'] = old_specs
        new_specs['type'] = 'changes_detector'
        return new_specs

    def __init__(self, storage, specifications):
        self._storage = storage
        self._specifications = specifications
        if self._specifications['type'] == 'limit_steps':
            self.get = self._limit_steps
        elif self._specifications['type'] == 'exponential_decay':
            self.get = self._exponential_decay
        elif self._specifications['type'] == 'fixed':
            self.get = self._fixed
        elif self._specifications['type'] == 'periodic_truth':
            self.get = self._periodic_truth
        elif self._specifications['type'] == 'true_on_steps':
            self.get = self._true_on_steps
        elif self._specifications['type'] == 'always_false':
            self.get = self._always_false
        elif self._specifications['type'] == 'changes_detector':
            self._value_controllers = list()
            self._last_values = list()
            for value_specs in self._specifications['old_specs']:
                self._value_controllers.append(Controller(self._storage, value_specs))
                self._last_values.append(self._value_controllers[-1].get())
            self.get = self._changes_detector

    def _changes_detector(self):
        something_changed = False
        for idx, (last_value, controller) in enumerate(zip(self._last_values, self._value_controllers)):
            if last_value != controller.get():
                something_changed = something_changed or True
                self._last_values[idx] = controller.get()
        return something_changed

    def _exponential_decay(self):
        num_stairs = self._storage['step'] // self._specifications['period']
        returned_value = self._specifications['init']
        return returned_value * self._specifications['decay']**num_stairs

    def _limit_steps(self):
        if self._storage['step'] > self._specifications['limit']:
            return False
        else:
            return True

    def _fixed(self):
        return self._specifications['value']

    def _periodic_truth(self):
        if self._storage['step'] % self._specifications['period'] == 0:
            return True
        else:
            return False

    def _true_on_steps(self):
        if self._storage['step'] in self._specifications['steps']:
            return True
        else:
            return False

    def _always_false(self):
        return False

    @property
    def name(self):
        return self._specifications['name']


class Handler(object):

    _stars = '*'*30

    def __init__(self,
                 environment_instance,
                 hooks,
                 processing_type,
                 save_path,
                 result_types,
                 summary=False,
                 add_graph_to_summary=False,
                 save_to_file=None,
                 save_to_storage=None,
                 print_results=None,
                 batch_generator_class=None,
                 vocabulary=None,
                 # several_launches method specific
                 eval_dataset_names=None,
                 hyperparameters=None,
                 # test method specific
                 validation_dataset_names=None,
                 validation_tensor_schedule=None,
                 printed_result_types=['loss'],
                 fuses=None,
                 fuse_tensor_schedule=None,
                 fuse_file_name=None):
        continuous_chit_chat = ['simple_fontain']
        self._processing_type = processing_type
        self._environment_instance = environment_instance
        self._save_path = save_path
        self._result_types = self._environment_instance.put_result_types_in_correct_order(result_types)
        self._bpc = 'bpc' in self._result_types
        self._hooks = hooks
        self._last_run_tensor_order = dict()
        self._save_to_file = save_to_file
        self._save_to_storage = save_to_storage
        self._print_results = print_results

        self._batch_generator_class = batch_generator_class
        self._vocabulary = vocabulary
        if self._save_path is not None:
            create_path(self._save_path)
        if self._processing_type == 'train':
            self._train_files = dict()
            if self._save_path is not None:
                self._train_files['loss'] = open(self._save_path +
                                                 '/' +
                                                 'loss_train.txt',
                                                 'a')
                self._train_files['perplexity'] = open(self._save_path +
                                                       '/' +
                                                       'perplexity_train.txt',
                                                       'a')
                self._train_files['accuracy'] = open(self._save_path +
                                                     '/' +
                                                     'accuracy_train.txt',
                                                     'a')
                if self._bpc:
                    self._train_files['bpc'] = open(self._save_path +
                                                    '/' +
                                                    'bpc_train.txt',
                                                    'a')
                self._train_files['pickle_tensors'] = open(self._save_path +
                                                           '/' +
                                                           'tensors_train.pickle',
                                                           'ab')
            self._train_dataset_name = None
            self._dataset_specific = dict()
            self._controllers = None
            self._results_collect_interval = None
            self._print_per_collected = None
            self._example_per_print = None
            self._train_tensor_schedule = None
            self._validation_tensor_schedule = None

            self._fuses = None
            self._print_fuses = True
            self._fuse_file_name = None
            self._fuse_tensor_schedule = None

            self._processed_fuse_index = None

            self._text_is_being_accumulated = False
            self._accumulated_text = None

            self._printed_result_types = None
            self._printed_controllers = None
            if summary and self._save_path is not None:
                self._writer = tf.summary.FileWriter(self._save_path + '/' + 'summary')
                if add_graph_to_summary:
                    self._writer.add_graph(tf.get_default_graph())
            self._environment_instance.init_storage('train',
                                                    steps=list(),
                                                    loss=list(),
                                                    perplexity=list(),
                                                    accuracy=list(),
                                                    bpc=list())
            self._training_step = None
            self._accumulation_is_performed = False
            self._accumulated_tensors = dict()
            self._accumulated = dict(loss=None, perplexity=None, accuracy=None)
            if self._bpc:
                self._accumulated['bpc'] = None
        if self._processing_type == 'test':
            self._name_of_dataset_on_which_accumulating = None
            self._dataset_specific = dict()
            self._validation_tensor_schedule = validation_tensor_schedule
            self._validation_dataset_names = validation_dataset_names
            self._switch_datasets(validation_dataset_names=self._validation_dataset_names)
            self._printed_result_types = printed_result_types
            self._accumulated_tensors = dict()
            self._accumulated = dict(loss=None, perplexity=None, accuracy=None)
            if self._bpc:
                self._accumulated['bpc'] = None
            self._fuses = fuses
            if self._fuses is not None:
                for fuse in self._fuses:
                    fuse['results'] = list()
            if fuse_file_name is not None:
                self._fuse_file_name = fuse_file_name
            elif self._fuses is not None and self._save_path is not None:
                self._fuse_file_name = add_index_to_filename_if_needed(self._save_path + '/fuses.txt')
            self._fuse_tensor_schedule = fuse_tensor_schedule

            self._processed_fuse_index = None

            if self._print_results is None:
                self._print_fuses = False
            else:
                self._print_fuses = self._print_results

            self._text_is_being_accumulated = False
            self._accumulated_text = None

        if self._processing_type == 'several_launches':
            self._result_types = result_types
            self._eval_dataset_names = eval_dataset_names
            self._save_path = save_path
            self._environment_instance = environment_instance
            self._hyperparameters = hyperparameters
            self._order = list(self._result_types)
            if self._hyperparameters is not None:
                self._order.extend(self._hyperparameters)
            create_path(self._save_path)
            self._file_descriptors = dict()
            self._tmpl = '%s '*(len(self._order) - 1) + '%s\n'
            for dataset_name in eval_dataset_names:
                self._file_descriptors[dataset_name] = open(self._save_path + '/' + dataset_name + '.txt',
                                                            'a')
                self._file_descriptors[dataset_name].write(self._tmpl % tuple(self._order))
            self._environment_instance.set_in_storage(launches=list())

        # The order in which tensors are presented in the list returned by get_additional_tensors method
        # It is a list. Each element is either tensor alias or a tuple if corresponding hook is pointing to a list of
        # tensors. Such tuple contains tensor alias, and sizes of nested lists

    def _switch_datasets(self, train_dataset_name=None, validation_dataset_names=None):
        if train_dataset_name is not None:
            self._train_dataset_name = train_dataset_name
        if validation_dataset_names is not None:
            #print('validation_dataset_names:', validation_dataset_names)
            #print('env._storage:', self._environment_instance._storage)
            for dataset_name in validation_dataset_names:
                if dataset_name not in self._dataset_specific.keys():
                    new_files = dict()
                    if self._save_path is not None:
                        new_files['loss'] = open(self._save_path +
                                                 '/' +
                                                 'loss_validation_%s.txt' % dataset_name,
                                                 'a')
                        new_files['perplexity'] = open(self._save_path +
                                                       '/' +
                                                       'perplexity_validation_%s.txt' % dataset_name,
                                                       'a')
                        new_files['accuracy'] = open(self._save_path +
                                                     '/' +
                                                     'accuracy_validation_%s.txt' % dataset_name,
                                                     'a')
                        if self._bpc:
                            new_files['bpc'] = open(self._save_path +
                                                    '/' +
                                                    'bpc_validation_%s.txt' % dataset_name,
                                                    'a')
                        new_files['pickle_tensors'] = open(self._save_path +
                                                           '/' +
                                                           'tensors_validation_%s.pickle' % dataset_name,
                                                           'ab')

                    self._dataset_specific[dataset_name] = {'name': dataset_name,
                                                            'files': new_files}
                    init_dict = dict()
                    for key in self._result_types:
                        if not self._environment_instance.check_if_key_in_storage([dataset_name, key]):
                            init_dict[key] = list()
                    #print('dataset_name:', dataset_name)
                    self._environment_instance.init_storage(dataset_name, **init_dict)
            for key in self._dataset_specific.keys():
                if key not in validation_dataset_names:
                    for file_d in self._dataset_specific[key]['files'].values():
                        file_d.close()
                    del self._dataset_specific[key]

    def set_new_run_schedule(self, schedule, train_dataset_name, validation_dataset_names):
        self._results_collect_interval = schedule['to_be_collected_while_training']['results_collect_interval']
        if self._results_collect_interval is not None:
            if self._result_types is not None:
                self._save_to_file = True
                self._save_to_storage = True
            else:
                self._save_to_file = False
                self._save_to_storage = False
        else:
            self._save_to_file = False
            self._save_to_storage = False
        self._print_per_collected = schedule['to_be_collected_while_training']['print_per_collected']
        self._example_per_print = schedule['to_be_collected_while_training']['example_per_print']
        if self._example_per_print is not None:
            self._print_fuses = True
        else:
            self._print_fuses = False
        self._train_tensor_schedule = schedule['train_tensor_schedule']
        self._validation_tensor_schedule = schedule['validation_tensor_schedule']
        self._printed_controllers = schedule['printed_controllers']
        self._printed_result_types = schedule['printed_result_types']

        self._fuses = schedule['fuses']
        self._fuse_tensor_schedule = schedule['fuse_tensors']
        if self._fuses is not None:
            for fuse in self._fuses:
                fuse['results'] = list()
        if self._fuse_file_name is None and self._fuses is not None and self._save_path is not None:
            self._fuse_file_name = add_index_to_filename_if_needed(self._save_path + '/fuses.txt')

        if self._printed_result_types is not None:
            if len(self._printed_result_types) > 0:
                self._print_results = True
        self._switch_datasets(train_dataset_name, validation_dataset_names)

    def set_test_specs(self, validation_dataset_names=None, fuses=None, replicas=None, random=None):
        if validation_dataset_names is not None:
            self._switch_datasets(validation_dataset_names=validation_dataset_names)

    def set_controllers(self, controllers):
        self._controllers = controllers

    def start_accumulation(self, dataset_name, training_step=None):
        self._name_of_dataset_on_which_accumulating = dataset_name
        self._training_step = training_step
        for res_type in self._accumulated.keys():
            self._accumulated[res_type] = list()

    @staticmethod
    def decide(higher_bool, lower_bool):
        if higher_bool is None:
            if lower_bool is None:
                answer = False
            else:
                answer = lower_bool
        else:
            answer = higher_bool
        return answer

    def stop_accumulation(self,
                          save_to_file=True,
                          save_to_storage=True,
                          print_results=True):
        save_to_file = self.decide(save_to_file, self._save_to_file)
        save_to_storage = self.decide(save_to_storage, self._save_to_storage)
        print_results = self.decide(print_results, self._print_results)
        means = dict()
        for key, value_list in self._accumulated.items():
            #print('%s:' % key, value_list)
            counter = 0
            mean = 0
            for value in value_list:
                if value >= 0.:
                    mean += value
                    counter += 1
            if counter == 0:
                mean = 0.
            else:
                mean = mean / counter
            if self._save_path is not None:
                if save_to_file:
                    file_d = self._dataset_specific[self._name_of_dataset_on_which_accumulating]['files'][key]
                    if self._training_step is not None:
                        file_d.write('%s %s\n' % (self._training_step, mean))
                    else:
                        file_d.write('%s\n' % (sum(value_list) / len(value_list)))
            means[key] = mean
        if save_to_storage:
            self._environment_instance.append_to_storage(self._name_of_dataset_on_which_accumulating,
                **dict([(key, means[key]) for key in self._result_types]))
        if print_results:
            self._print_standard_report(
                regime='validation',
                message='results on validation dataset %s' % self._name_of_dataset_on_which_accumulating,
                **means)
        self._training_step = None
        self._name_of_dataset_on_which_accumulating = None
        self._save_accumulated_tensors()
        return means

    def set_processed_fuse_index(self, fuse_idx):
        self._processed_fuse_index = fuse_idx

    def start_text_accumulation(self):
        self._accumulated_text = ''
        self._text_is_being_accumulated = True

    def stop_text_accumulation(self):
        print('self._fuses:', self._fuses)
        self._fuses[self._processed_fuse_index]['results'].append(str(self._accumulated_text))
        self._accumulated_text = None
        self._text_is_being_accumulated = False

    def _prepair_string(self, res):
        preprocessed = ''
        for char in res:
            preprocessed += self._form_string_char(char)
        return preprocessed

    def _form_fuse_msg(self, training_step):
        msg = ''
        if training_step is not None:
            msg += 'generation from fuses on step %s' % str(training_step) + '\n'
        else:
            msg += 'generation from fuses\n'
        msg += (self._stars + '\n') * 2
        for fuse in self._fuses:
            msg += ('\nfuse: ' + fuse['text'] + '\n')
            msg += self._stars + '\n'
            for res_idx, res in enumerate(fuse['results']):
                if res_idx > 0:
                    msg += ('\nlaunch number: ' + str(res_idx) + '\n')
                else:
                    msg += ('launch number: ' + str(res_idx) + '\n')
                msg += ('result: ' + self._prepair_string(res) + '\n')
            msg += self._stars + '\n'*2
        msg += (self._stars + '\n') * 2
        return msg

    def _print_fuse_results(self, training_step):
        print(self._form_fuse_msg(training_step))

    def _save_fuse_results(self, training_step):
        with open(self._fuse_file_name, 'a', encoding='utf-8') as fd:
            fd.write(self._form_fuse_msg(training_step) + '\n'*2)

    def clean_fuse_results(self):
        for fuse in self._fuses:
            fuse['results'] = list()

    def dispense_fuse_results(self, training_step):
        if self._print_fuses:
            self._print_fuse_results(training_step)
        if self._save_path is not None:
            self._save_fuse_results(training_step)
        for fuse in self._fuses:
            fuse['results'] = list()

    def _process_validation_results(self,
                                    step,
                                    validation_res):
        if self._bpc:
            [loss, perplexity, accuracy, bpc] = validation_res[1:5]
        else:
            [loss, perplexity, accuracy] = validation_res[1:4]
        if self._bpc:
            self._accumulate_several_data(['loss', 'perplexity', 'accuracy', 'bpc'], [loss, perplexity, accuracy, bpc])
            self._accumulate_tensors(step, validation_res[5:])
        else:
            self._accumulate_several_data(['loss', 'perplexity', 'accuracy'], [loss, perplexity, accuracy])
            self._accumulate_tensors(step, validation_res[4:])

    def _cope_with_tensor_alias(self,
                                alias):
        if not isinstance(self._hooks[alias], list):
            return [self._hooks[alias]], 1
        add_tensors = list()
        order = [alias, len(self._hooks[alias])]
        counter = 0
        if isinstance(self._hooks[alias][0], list):
            order.append(len(self._hooks[alias][0]))
            for elem in self._hooks[alias]:
                for tensor in elem:
                    add_tensors.append(tensor)
                    counter += 1
        else:
            for tensor in self._hooks[alias]:
                add_tensors.append(tensor)
                counter += 1
        return add_tensors, counter

    def _save_datum(self, descriptor, step, datum, processing_type, dataset_name):
        if processing_type == 'train':
            self._train_files[descriptor].write('%s %s\n' % (step, datum))
        elif processing_type == 'validation':
            self._dataset_specific[dataset_name]['files'][descriptor].write('%s %s\n' % (step, datum))

    def _save_launch_results(self, results, hp):
        for dataset_name, res in results.items():
            values = list()
            all_together = dict(hp.items())
            #print('dataset_name:', dataset_name)
            #print('all_together:', all_together)
            all_together.update(res)
            for key in self._order:
                values.append(all_together[key])
            self._file_descriptors[dataset_name].write(self._tmpl % tuple(values))

    def _save_several_data(self,
                           descriptors,
                           step,
                           data,
                           processing_type='train',
                           dataset_name=None):
        for descriptor, datum in zip(descriptors, data):
            if datum is not None:
                self._save_datum(descriptor, step, datum, processing_type, dataset_name)

    def _save_accumulated_tensors(self):
        pass

    def _accumulate_several_data(self, descriptors, data):
        for descriptor, datum in zip(descriptors, data):
            if datum is not None:
                self._accumulated[descriptor].append(datum)

    def get_tensors(self, regime, step, with_assistant=False):
        tensors = list()
        self._last_run_tensor_order = dict()
        pointer = 0
        current = dict()
        self._last_run_tensor_order['basic'] = current
        current['tensors'] = dict()
        start = pointer
        if regime == 'train':
            if with_assistant:
                tensors.append(self._hooks['train_op_with_assistant'])
                current['tensors']['train_op_with_assistant'] = [pointer, pointer+1]
                pointer += 1
            else:
                tensors.append(self._hooks['train_op'])
                current['tensors']['train_op'] = [pointer, pointer + 1]
                pointer += 1
            for res_type in self._result_types:
                tensors.append(self._hooks[res_type])
                current['tensors'][res_type] = [pointer, pointer + 1]
                pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]

            if self._train_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._train_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'validation':
            tensors.append(self._hooks['validation_predictions'])
            current['tensors']['validation_predictions'] = [pointer, pointer + 1]
            pointer += 1
            for res_type in self._result_types:
                tensors.append(self._hooks['validation_' + res_type])
                current['tensors']['validation_' + res_type] = [pointer, pointer + 1]
                pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]

            if self._validation_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._validation_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'fuse':
            tensors.append(self._hooks['validation_predictions'])
            current['tensors']['validation_predictions'] = [pointer, pointer + 1]
            pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]
            if self._fuse_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._fuse_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)

        #print(tensors)
        return tensors

    def _get_additional_tensors(self,
                                schedule,
                                step,
                                start_pointer):
        #print('_get_additional_tensors method. schedule:', schedule)
        additional_tensors = list()
        pointer = start_pointer
        for tensors_use, tensors_schedule in schedule.items():
            #print('tensor_schedule:', tensors_schedule)
            self._last_run_tensor_order[tensors_use] = dict()
            self._last_run_tensor_order[tensors_use]['tensors'] = dict()
            start = pointer
            if isinstance(tensors_schedule, dict):
                for tensor_alias, tensor_schedule in tensors_schedule.items():
                    if isinstance(tensor_schedule, list):
                        if step in tensor_schedule:
                            add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                            additional_tensors.extend(add_tensors)
                            self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                                 pointer + counter]
                            pointer += counter
                    elif isinstance(tensor_schedule, int):
                        if step % tensor_schedule == 0:
                            add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                            additional_tensors.extend(add_tensors)
                            self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                                 pointer + counter]
                            pointer += counter
            elif isinstance(tensors_schedule, list):
                for tensor_alias in tensors_schedule:
                    add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                    additional_tensors.extend(add_tensors)
                    self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                         pointer + counter]
                    pointer += counter
            self._last_run_tensor_order[tensors_use]['borders'] = [start, pointer]
        return additional_tensors

    def _print_tensors(self, instructions, print_step_number=False, indent=0):
        if print_step_number:
            print('\n'*indent + 'step:', instructions['step'])
        print(instructions['message'])
        for alias, res in instructions['results'].items():
            if not isinstance(res, list):
                print('%s:' % alias, res)
            else:
                print('%s:' % alias)
                if isinstance(res[0], list):
                    for high_idx, high_elem in res.items():
                        print('\n\n[%s]:' % high_idx)
                        for low_idx, low_elem in high_elem.items():
                            print('\n'*4 + '[%s][%s]:' % (high_idx, low_idx), low_elem)
                else:
                    for idx, elem in res.items():
                        print('\n\n[%s]:' % idx, elem)

    def _print_launch_results(self, results, hp, idx=None, indent=2):
        if indent != 0:
            print('\n' * (indent - 1))
        for key in self._order:
            if key in hp:
                print('%s: %s' % (key, hp[key]))
        for dataset_name, res in results.items():
            print('results on %s dataset:' % dataset_name)
            for key in self._order:
                if key in res:
                    print('%s: %s' % (key, res[key]))

    def _accumulate_tensors(self, step, tensors):
        pass

    def _save_tensors(self, tensors):
        pass

    def _print_controllers(self):
        if self._controllers is not None:
            for controller in self._controllers:
                # if isinstance(controller, Controller):
                #     print(controller.name)
                # if isinstance(controller, list):
                #     for c in controller:
                #         if isinstance(c, Controller):
                #             print(c.name)
                #         else:
                #             print(c)
                #print('controller._specifications:', controller._specifications)
                if controller.name in self._printed_controllers:
                    print('%s:' % controller.name, controller.get())

    def _print_standard_report(self,
                               indents=[0, 0],
                               regime='train',
                               **kwargs):
        for _ in range(indents[0]):
            print('')
        if regime == 'train':
            if 'step' in kwargs:
                print('step:', kwargs['step'])
            self._print_controllers()
        if 'message' in kwargs:
            print(kwargs['message'])
        for key, value in kwargs.items():
            if key != 'tensors' and key != 'step' and key != 'message' and key in self._printed_result_types:
                print('%s:' % key, value)
        if 'tensors' in kwargs:
            self._print_tensors(kwargs['tensors'], self._train_tensor_schedule)
        for _ in range(indents[1]):
            print('')

    def _get_structure_of_hook(self, alias):
        if not isinstance(self._hooks[alias], list):
            return 1
        else:
            if not isinstance(self._hooks[alias][0], list):
                return [len(self._hooks[alias])]
            else:
                output = [len(self._hooks[alias])]
                for l in self._hooks[alias]:
                    output.append(len(l))
                return output

    def _extract_results(self, last_order, tensor_use, train_res):
        extracted = dict()
        for alias, borders in last_order[tensor_use]['tensors'].items():
            structure = self._get_structure_of_hook(alias)
            if isinstance(structure, int):
                extracted[alias] = train_res[borders[0]]
            elif isinstance(structure, list):
                if len(structure) == 1:
                    extracted[alias] = train_res[borders[0], borders[1]]
                else:
                    structured = list()
                    pointer = borders[0]
                    for length in structure[1:]:
                        structured.append(train_res[pointer, pointer+length])
                    extracted[alias] = structured
        return extracted

    def _form_train_tensor_print_instructions(self, step, train_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'train tensors:'
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'train_print_tensors', train_res)
        #print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _process_train_results(self,
                               step,
                               train_res):
        # print('step:', step)
        # print('train_res:', train_res)
        #print(self._last_run_tensor_order)
        basic_borders = self._last_run_tensor_order['basic']['borders']
        tmp = train_res[basic_borders[0]+1:basic_borders[1]]
        if self._bpc:
            [loss, perplexity, accuracy, bpc] = tmp
        else:
            [loss, perplexity, accuracy] = tmp

        if self._printed_result_types is not None:
            if self._results_collect_interval is not None:
                if step % (self._results_collect_interval * self._print_per_collected) == 0:
                    if self._bpc:
                        self._print_standard_report(indents=[2, 0],
                                                    step=step,
                                                    loss=loss,
                                                    bpc=bpc,
                                                    perplexity=perplexity,
                                                    accuracy=accuracy,
                                                    message='results on train dataset')
                    else:
                        self._print_standard_report(indents=[2, 0],
                                                    step=step,
                                                    loss=loss,
                                                    perplexity=perplexity,
                                                    accuracy=accuracy,
                                                    message='results on train dataset')
        if self._results_collect_interval is not None:
            if step % self._results_collect_interval == 0:
                if self._bpc:
                    if self._save_path is not None:
                        self._save_several_data(['loss', 'perplexity', 'accuracy', 'bpc'],
                                                step,
                                                [loss, perplexity, accuracy, bpc])
                    self._environment_instance.append_to_storage('train',
                                                                 loss=loss,
                                                                 bpc=bpc,
                                                                 perplexity=perplexity,
                                                                 accuracy=accuracy,
                                                                 steps=step)
                else:
                    if self._save_path is not None:
                        self._save_several_data(['loss', 'perplexity', 'accuracy'], step, [loss, perplexity, accuracy])
                    self._environment_instance.append_to_storage('train',
                                                                 loss=loss,
                                                                 perplexity=perplexity,
                                                                 accuracy=accuracy,
                                                                 steps=step)
        print_borders = self._last_run_tensor_order['train_print_tensors']['borders']
        if print_borders[1] - print_borders[0] > 0:
            print_instructions = self._form_train_tensor_print_instructions(step,
                                                                            train_res,
                                                                            self._last_run_tensor_order)
            other_stuff_is_printed = (step % (self._results_collect_interval * self._print_per_collected) == 0)
            if other_stuff_is_printed:
                indent = 0
            else:
                indent = 1
            self._print_tensors(print_instructions,
                                print_step_number=not other_stuff_is_printed,
                                indent=indent)

    @staticmethod
    def _form_string_char(char):
        special_characters_map = {'\n': '\\n',
                                  '\t': '\\t'}
        if char in list(special_characters_map.keys()):
            return special_characters_map[char]
        else:
            return char

    def _form_fuse_tensor_print_instructions(self, step, char, fuse_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'fuse tensors:\nchar = %s' % self._form_string_char(char)
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'fuse_print_tensors', fuse_res)
        #print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _process_char_generation_results(self, step, res):
        basic_borders = self._last_run_tensor_order['basic']['borders']
        [prediction] = res[basic_borders[0]:basic_borders[1]]
        char = self._batch_generator_class.vec2char(prediction, self._vocabulary)[0]
        if self._text_is_being_accumulated:
            self._accumulated_text += char
        if 'fuse_print_tensors' in self._last_run_tensor_order:
            print_borders = self._last_run_tensor_order['fuse_print_tensors']['borders']
            if print_borders[1] - print_borders[0] > 0:
                print_instructions = self._form_fuse_tensor_print_instructions(step,
                                                                               char,
                                                                               res,
                                                                               self._last_run_tensor_order)
                self._print_tensors(print_instructions,
                                    print_step_number=True,
                                    indent=1)

    def _several_launches_results_processing(self, hp, results):
        self._environment_instance.set_in_storage(launches=(results, hp))
        self._save_launch_results(results, hp)
        self._print_launch_results(results, hp)

    def process_results(self, *args, regime=None):
        if regime == 'train':
            step = args[0]
            res = args[1]
            self._process_train_results(step, res)
        if regime == 'validation':
            step = args[0]
            res = args[1]
            self._process_validation_results(step, res)
        if regime == 'fuse':
            step = args[0]
            res = args[1]
            self._process_char_generation_results(step, res)
        if regime =='several_launches':
            hp = args[0]
            res = args[1]
            self._several_launches_results_processing(hp, res)

    def close(self):
        for file in self._train_files.values():
            file.close()
        for dataset in self._dataset_specific.values():
            for file_d in dataset['files'].values():
                file_d.close()


class InvalidArgumentError(Exception):
    def __init__(self, msg, value, name, allowed_values):
        super(InvalidArgumentError, self).__init__(msg)
        self._msg = msg
        self._value = value
        self._name = name
        self._allowed_values = allowed_values


def perplexity_tensor(**kwargs):
    probabilities = kwargs['probabilities']
    labels = kwargs['labels']
    special_args = kwargs['special_args']
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            probabilities_shape = probabilities.get_shape().as_list()
            length = probabilities_shape[1]
            _, switch = tf.split(labels, [length, 1], axis=1)
            switch = tf.reshape(switch, [-1])
    ln2 = np.log(2)
    shape = probabilities.get_shape().as_list()
    probabilities = tf.where(probabilities > 1e-10, probabilities, np.full(tuple(shape), 1e-10))
    log_probabilities = tf.log(probabilities) / ln2
    entropy = tf.reduce_sum(- probabilities * log_probabilities, axis=1)
    perplexity = tf.exp(ln2 * entropy)
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            perplexity = perplexity * switch
            num_of_sensible_results = tf.reduce_sum(switch)
            there_is_sensible = tf.not_equal(num_of_sensible_results, 0.)
            mean_perplexity = tf.divide(tf.reduce_sum(perplexity, name='sum_perplexity'),
                                        (num_of_sensible_results + 1e-12),
                                        name='mean_perplexity')
            return tf.where(there_is_sensible, mean_perplexity, -1.)
    return tf.reduce_mean(perplexity, name="mean_perplexity")


def loss_tensor(**kwargs):
    predictions = kwargs['predictions']
    labels = kwargs['labels']
    special_args = kwargs['special_args']
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            predictions_shape = predictions.get_shape().as_list()
            length = predictions_shape[1]
            labels, switch = tf.split(labels, [length, 1], axis=1)
            switch = tf.reshape(switch, [-1])
    shape = predictions.get_shape().as_list()
    predictions = tf.where(predictions > 1e-10, predictions, np.full(tuple(shape), 1e-10))
    log_predictions = tf.log(predictions)
    loss_on_characters = tf.reduce_sum(-labels * log_predictions, axis=1)
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            loss_on_characters = loss_on_characters * switch
            num_of_sensible_results = tf.reduce_sum(switch)
            there_is_sensible = tf.not_equal(num_of_sensible_results, 0.)
            mean_loss = tf.reduce_sum(loss_on_characters, name='mean_perplexity') / (num_of_sensible_results + 1e-12)
            return tf.where(there_is_sensible, mean_loss, -1.)
    return tf.reduce_mean(loss_on_characters)


def bpc_tensor(**kwargs):
    return kwargs['loss'] / np.log(2)


def accuracy_tensor(**kwargs):
    predictions = kwargs['predictions']
    labels = kwargs['labels']
    special_args = kwargs['special_args']
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            predictions_shape = predictions.get_shape().as_list()
            length = predictions_shape[1]
            labels, switch = tf.split(labels, [length, 1], axis=1)
            switch = tf.reshape(switch, [-1])
    predictions = tf.argmax(predictions, axis=1)
    labels = tf.argmax(labels, axis=1)
    accuracy = tf.to_float(tf.equal(predictions, labels))
    if 'dialog_switch' in special_args:
        if special_args['dialog_switch']:
            accuracy = accuracy * switch
            num_of_sensible_results = tf.reduce_sum(switch)
            there_is_sensible = tf.not_equal(num_of_sensible_results, 0.)
            mean_accuracy = tf.reduce_sum(accuracy, name='mean_perplexity') / (num_of_sensible_results + 1e-12)
            return tf.where(there_is_sensible, mean_accuracy, -1.)
    return tf.reduce_mean(accuracy)


def identity_tensor(**kwargs):
    if len(kwargs) > 1:
        raise InvalidArgumentError('kwargs should not contain 1 entry', kwargs, 'kwargs', 'len(kwargs)=1')
    for value in kwargs.values():
        return value


class Environment(object):

    @staticmethod
    def put_result_types_in_correct_order(result_types):
        correct_order = ['loss', 'perplexity', 'accuracy', 'bpc']
        sorted_types = list()
        for result_type in correct_order:
            if result_type in result_types:
                sorted_types.append(result_type)
        return sorted_types

    def __init__(self,
                 pupil_class,
                 batch_generator_classes,
                 vocabulary=None,
                 datasets=None,
                 filenames=None,
                 texts=None,
                 assistant_class=None):
        """ Initializes environment class
        Args:
            pupil_class: is a class to which pupil model belongs
            assistant_class: is a class to which assistant model belongs if it is provided
            data_filenames: contains paths to a files with data for model training, validation and testing
                has to be a dictionary in which keys are names of datasets, values are strings with paths to files
            batch_generator_classes: """

        self._pupil_class = pupil_class
        self._pupil_type = self._pupil_class.get_name()
        self._assistant_class = assistant_class

        if datasets is not None:
            self._datasets = dict()
            for dataset in datasets:
                self._datasets[dataset[1]] = dataset
        else:
            self._datasets = dict()

        self._vocabulary = vocabulary

        if filenames is not None:
            for filename in filenames:
                key, value = self._process_dataset_filename(filename)
                self._datasets[key] = [value, key]

        if texts is not None:
            for text in texts:
                key, value = self._process_input_text_dataset(text)
                self._datasets[key] = [value, key]

        if not isinstance(batch_generator_classes, dict):
            self._batch_generator_classes = {'default': batch_generator_classes}
        else:
            self._batch_generator_classes = batch_generator_classes

        # # Just initializing attributes containing arguments for model building
        # self._pupil_building_parameters = self._pupil_class.get_building_parameters()
        # if self._assistant_class is not None:
        #     self._assistant_building_parameters = self._assistant_class.get_building_parameters()

        # An attributes containing instance of self._model_class. While graph is not built self._model is None
        self._pupil = None
        self._assistant = None

        # An attribute holding tensors which could be run. It has the form of dictionary which keys are user specified
        # descriptors of tensors and are tensors themselves
        self._pupil_hooks = dict()
        self._assistant_hooks = dict()

        # List containing fuses. They are used for testing the model. You may feed them to the model and see how it
        # continues generating after that
        self._fuses = list()

        # An attribute holding session. Default value when there is no active sessions is None
        self._session = None

        self._build_functions = {'identity': identity_tensor}

        pupil_special_args = self._pupil_class.get_special_args()
        train_perplexity_builder = dict(f=perplexity_tensor,
                                        hooks={'probabilities': 'predictions',
                                               'labels': 'labels'},
                                        tensor_names=dict(),
                                        output_hook_name='perplexity',
                                        special_args=pupil_special_args)
        valid_perplexity_builder = dict(f=perplexity_tensor,
                                        hooks={'probabilities': 'validation_predictions',
                                               'labels': 'validation_labels'},
                                        tensor_names=dict(),
                                        output_hook_name='validation_perplexity',
                                        special_args=pupil_special_args)
        valid_loss_builder = dict(f=loss_tensor,
                                  hooks={'predictions': 'validation_predictions',
                                         'labels': 'validation_labels'},
                                  tensor_names=dict(),
                                  output_hook_name='validation_loss',
                                  special_args=pupil_special_args)
        train_bpc_builder = dict(f=bpc_tensor,
                                 hooks={'loss': 'loss'},
                                 tensor_names=dict(),
                                 output_hook_name='bpc',
                                 special_args=pupil_special_args)
        valid_bpc_builder=dict(f=bpc_tensor,
                               hooks={'loss': 'validation_loss'},
                               tensor_names=dict(),
                               output_hook_name='validation_bpc',
                               special_args=pupil_special_args)
        train_accuracy_builder=dict(f=accuracy_tensor,
                                    hooks={'predictions': 'predictions',
                                          'labels': 'labels'},
                                    tensor_names=dict(),
                                    output_hook_name='accuracy',
                                    special_args=pupil_special_args)
        valid_accuracy_builder=dict(f=accuracy_tensor,
                                    hooks={'predictions': 'validation_predictions',
                                           'labels': 'validation_labels'},
                                    tensor_names=dict(),
                                    output_hook_name='validation_accuracy',
                                    special_args=pupil_special_args)

        self._builders = {'perplexity': train_perplexity_builder,
                          'validation_perplexity': valid_perplexity_builder,
                          'validation_loss': valid_loss_builder,
                          'bpc': train_bpc_builder,
                          'validation_bpc': valid_bpc_builder,
                          'accuracy': train_accuracy_builder,
                          'validation_accuracy': valid_accuracy_builder}

        tensor_schedule = {'train_print_tensors': dict(),
                           'train_save_tensors': dict(),
                           'train_print_text_tensors': dict(),
                           'train_save_text_tensors': dict(),
                           'train_summary_tensors': dict()}

        valid_tensor_schedule = {'valid_print_tensors': dict(),
                                 'valid_save_text_tensors': dict()}

        fuse_tensors = {'fuse_print_tensors': dict(), 'fuse_save_tensors': dict()}

        # Every results_collect_interval-th step BPC, accuracy, perplexity are collected
        # Every print_per_collected-th point containing BPC, accuracy and perplexity is printed
        # Together with every example_per_print-th point example is printed
        default_collected_while_training = {'results_collect_interval': 100,
                                            'print_per_collected': 1,
                                            'example_per_print': 1}

        default_collected_on_validation = {}

        default_learning_rate_control = {'init': 0.002,
                                         'decay': 0.8,
                                         'period': 1000,
                                         'type': 'exponential_decay',
                                         'name': 'learning_rate'}

        if len(self._datasets) > 0:
            default_dataset = self._datasets[0]
        else:
            default_dataset = None
        _, gens = zip(*sorted(self._batch_generator_classes.items()))
        self._default_batch_generator = gens[0]
        self._default_train_method_args = dict(
            session_specs={'allow_soft_placement': False,
                           'gpu_memory': None,
                           'log_device_placement': False,
                           'visible_device_list': ""},
            start_specs={'restore_path': None,
                         'save_path': None,
                         'result_types': self.put_result_types_in_correct_order(
                             ['loss', 'perplexity', 'accuracy']),
                         'summary': False,
                         'add_graph_to_summary': False,
                         'batch_generator_class': self._default_batch_generator,
                         'vocabulary': self._vocabulary},
            run=dict(
                train_specs={'assistant': None,
                             'learning_rate': construct(default_learning_rate_control),
                             'additions_to_feed_dict': None,
                             'stop': {'type': 'limit_steps', 'limit': 10000, 'name': 'stop'},
                             'train_dataset': default_dataset,
                             'batch_size': {'type': 'fixed', 'value': 64, 'name': 'batch_size'},
                             'train_batch_kwargs': dict(),
                             'checkpoint_steps': None,
                             'debug': None,
                             'validation_datasets': None,
                             'validation_batch_size': 1,
                             'valid_batch_kwargs': dict(),
                             'no_validation': False},
                schedule={'to_be_collected_while_training': construct(default_collected_while_training),
                          'printed_result_types':  self.put_result_types_in_correct_order(
                             ['loss']),
                          'printed_controllers': ['learning_rate'],
                          'fuses': None,
                          'fuse_tensors': construct(fuse_tensors),
                          'replicas': None,
                          'random': {'number_of_runs': 5,
                                     'length': 80},
                          'train_tensor_schedule': construct(tensor_schedule),
                          'validation_tensor_schedule': construct(valid_tensor_schedule)}
                    )
                                               )
        self._default_test_method_args = dict(
            session_specs={'allow_soft_placement': False,
                           'gpu_memory': None,
                           'log_device_placement': False,
                           'visible_device_list': ""},
            start_specs={'restore_path': None,
                         'save_path': None,
                         'print_results': True,
                         'result_types': self.put_result_types_in_correct_order(
                             ['loss', 'perplexity', 'accuracy']),
                         'batch_generator_class': self._default_batch_generator,
                         'vocabulary': self._vocabulary},
            work=dict(additions_to_feed_dict=dict(),
                      debug=None,
                      validation_datasets=None,
                      validation_batch_size=1,
                      valid_batch_kwargs=dict(),
                      printed_result_types=self.put_result_types_in_correct_order(['loss']),
                      fuses=None,
                      fuse_tensors=construct(fuse_tensors),
                      fuse_file_name=None,
                      replicas=None,
                      random={'number_of_runs': 5,
                              'length': 80},
                      validation_tensor_schedule=construct(valid_tensor_schedule)
                    )
                                               )
        # This attribute is used solely for controlling learning parameters (learning rate, additions_to_feed_dict)
        # It is used by instances of Controller class
        # BPI stands for bits per input. It is cross entropy computed using logarithm for base 2
        self._handler = None
        self._storage = {'step': None}
        self._collected_result = None

    def build(self, **kwargs):
        """A method building the graph
        Args:
            kwargs: key word arguments passed to self._model_class constructor
            :type kwargs: dictionary"""

        # checking if passed required arguments
        self._build(kwargs)

    def _build(self, kwargs):
        self._pupil_class.check_kwargs(**kwargs)

        # Building the graph
        self._pupil = self._pupil_class(**kwargs)

        # getting default hooks
        default_hooks = self._pupil.get_default_hooks()
        self._pupil_hooks.update(default_hooks)

    def _split_to_loss_and_not_loss_names(self, names):
        loss_names = list()
        not_loss_names = list()
        for name in names:
            if 'loss' in name:
                loss_names.append(name)
            else:
                not_loss_names.append(name)
        return loss_names, not_loss_names

    def _arguments_for_new_tensor_building(self, hooks, tensor_names):
        arguments = dict()
        for key, value in hooks.items():
            if value not in self._pupil_hooks:
                stars = '\n**********\n'
                msg = "Warning! Adding to hooks shapeless placeholder of type tf.float32 with alias '%s'" % value
                print(stars + msg + stars)
                self._pupil_hooks[value] = tf.placeholder(tf.float32)
            arguments[key] = self._pupil_hooks[value]
        for key, value in tensor_names.items():
            arguments[key] = tf.get_default_graph().get_tensor_by_name(value)
        return arguments

    def _add_hook(self, builder_name, model_type='pupil'):
        if builder_name in self._builders:
            builder = self._builders[builder_name]
            kwargs = self._arguments_for_new_tensor_building(builder['hooks'],
                                                             builder['tensor_names'])
            kwargs['special_args'] = builder['special_args']
            new_tensor = builder['f'](**kwargs)
            if model_type == 'pupil':
                self._pupil_hooks[builder['output_hook_name']] = new_tensor
            else:
                self._assistant_hooks[builder['output_hook_name']] = new_tensor
        else:
            stars = '\n**********\n'
            msg = "Warning! Adding to hooks shapeless placeholder of type tf.float32 with alias '%s'" % builder_name
            print(stars + msg + stars)
            if model_type == 'pupil':
                self._pupil_hooks[builder_name] = tf.placeholder(tf.float32, name=builder_name)
            else:
                self._assistant_hooks[builder_name] = tf.placeholder(tf.float32, name=builder_name)

    def add_hooks(self, builder_names_or_builders=[], tensor_names=[], model_type='pupil'):
        actual_names = list()
        for builder_name in builder_names_or_builders:
            if isinstance(builder_name, dict):
                self.register_builder(**builder_name)
                actual_names.append(builder_name['output_hook_name'])
            else:
                actual_names.append(builder_name)
        loss_builder_names, not_loss_builder_names = self._split_to_loss_and_not_loss_names(actual_names)
        #print('loss_builder_names:', loss_builder_names)
        #print('not_loss_builder_names:', not_loss_builder_names)
        for builder_name in loss_builder_names:
            self._add_hook(builder_name, model_type=model_type)
        for builder_name in not_loss_builder_names:
            self._add_hook(builder_name, model_type=model_type)
        for alias, name in tensor_names:
            self._hooks[alias] = tf.get_default_graph().get_tensor_by_name(name)

    def register_build_function(self, function, name):
        self._build_functions[name] = function

    def print_available_builders(self):
        for builder_name, builder in self._builders.items():
            print(builder_name + ':', builder)

    def register_builder(self,
                         f=None,
                         hooks=dict(),
                         tensor_names=dict(),
                         output_hook_name=None,
                         special_args=None):
        if isinstance(f, str):
            f = self._build_functions[f]
        self._builders[output_hook_name] = dict(f=f,
                                                hooks=hooks,
                                                tensor_names=tensor_names,
                                                output_hook_name=output_hook_name,
                                                special_args=special_args)

    @classmethod
    def _update_dict(cls, dict_to_update, update):
        """Checks if update matches dict_to_update and updates it
        Args:
            dict_to_update: a class attribute of type dict which should be updated
            update: dict which is used for updating"""
        keys_all_right = match_two_dicts(update, dict_to_update)
        if keys_all_right:
            for key, value in update.items():
                if isinstance(value, dict):
                    cls._update_dict(dict_to_update[key], update[key])
                else:
                    dict_to_update[key] = construct(value)

    @property
    def default_train_method_args(self):
        return construct(self._default_train_method_args)

    @default_train_method_args.setter
    def default_train_method_args(self, update):
        """update is a dictionary which should match keys of self._pupil_default_training"""
        self._update_dict(self._default_train_method_args, update)

    @property
    def default_test_method_args(self):
        return construct(self._default_test_method_args)

    @default_test_method_args.setter
    def default_test_method_args(self, update):
        """update is a dictionary which should match keys of self._pupil_default_training"""
        self._update_dict(self._default_test_method_args, update)

    def get_default_method_parameters(self,
                                      method_name):
        if method_name == 'train':
            return self.default_train_method_args
        if method_name == 'test':
            return self.default_test_method_args
        return None

    def _start_session(self, allow_soft_placement, log_device_placement, gpu_memory, visible_device_list):
        """Starts new session with specified parameters. If there is opend session closes it"""
        if self._session is not None:
            print('Warning: there is an opened session already. Closing it')
            self._session.close()
        config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                log_device_placement=log_device_placement,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory,
                                                          visible_device_list=visible_device_list))
        self._session = tf.Session(config=config)

    def _close_session(self):
        self._session.close()
        self._session = None

    def init_storage(self, dataset_name, **kwargs):
        self._storage[dataset_name] = dict()
        d = self._storage[dataset_name]
        for key, value in kwargs.items():
            d[key] = value

    def append_to_storage(self, dataset_name, **kwargs):
        d = self._storage[dataset_name]
        for key, value in kwargs.items():
            d[key].append(value)

    def flush_storage(self):
        self._storage = {'step': None}

    def set_in_storage(self, **kwargs):
        for key, value in kwargs.items():
            self._storage[key] = value

    def check_if_key_in_storage(self, keys):
        return check_if_key_in_nested_dict(self._storage, keys)

    def _create_checkpoint(self, step, checkpoints_path, model_type='pupil'):
        path = checkpoints_path + '/' + str(step)
        print('\nCreating checkpoint at %s' % path)
        if model_type == 'pupil':
            self._pupil_hooks['saver'].save(self._session, path)
        elif model_type == 'assistant':
            self._assistant_hooks['saver'].save(self._session, path)

    def _initialize_pupil(self, restore_path):
        if restore_path is not None:
            self._pupil_hooks['saver'].restore(self._session, restore_path)
        else:
            self._session.run(tf.global_variables_initializer())

    def test(self,
             **kwargs):
        tmp_output = self._parse_1_set_of_kwargs(kwargs,
                                                 'test',
                                                 None,
                                                 False)
        all_tensor_aliases = self._all_tensor_aliases_from_test_method_arguments(tmp_output)
        print('all_tensor_aliases:', all_tensor_aliases)
        self._create_missing_hooks(all_tensor_aliases)
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        work = tmp_output['work']
        dataset_names = [dataset[1] for dataset in work['validation_datasets']]
        print("work['fuses']:", work['fuses'])
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['visible_device_list'])
        self._initialize_pupil(start_specs['restore_path'])
        add_feed_dict = dict()
        for tensor_alias, value in work['additions_to_feed_dict'].items():
            add_feed_dict[self._pupil_hooks[tensor_alias]] = value
        batch_generator_class = start_specs['batch_generator_class']
        self._handler = Handler(self,
                                self._pupil_hooks,
                                'test',
                                start_specs['save_path'],
                                start_specs['result_types'],
                                save_to_file=True,
                                save_to_storage=True,
                                print_results=start_specs['print_results'],
                                batch_generator_class=batch_generator_class,
                                vocabulary=start_specs['vocabulary'],
                                validation_dataset_names=dataset_names,
                                validation_tensor_schedule=work['validation_tensor_schedule'],
                                fuses=work['fuses'],
                                fuse_tensor_schedule=work['fuse_tensors'],
                                fuse_file_name=work['fuse_file_name'])

        empty_batch_gen = batch_generator_class('', 1, vocabulary=start_specs['vocabulary'])
        self._on_fuses(empty_batch_gen,
                       work['fuses'],
                       additional_feed_dict=add_feed_dict)

        validation_datasets = work['validation_datasets']
        for validation_dataset in validation_datasets:
            _ = self._validate(batch_generator_class,
                               validation_dataset,
                               work['validation_batch_size'],
                               work['valid_batch_kwargs'],
                               additional_feed_dict=add_feed_dict)

    def _on_fuses(self,
                  batch_generator,
                  fuses,
                  training_step=None,
                  additional_feed_dict=None):
        for fuse_idx, fuse in enumerate(fuses):
            self._handler.set_processed_fuse_index(fuse_idx)
            for repeat_idx in range(fuse['num_repeats']):
                if 'randomize_sample_state' in self._pupil_hooks:
                    self._session.run(self._pupil_hooks['randomize_sample_state'])
                elif 'reset_validation_state' in self._pupil_hooks:
                    self._session.run(self._pupil_hooks['reset_validation_state'])
                for char_idx, char in enumerate(fuse['text']):
                    vec = batch_generator.char2vec(char, batch_generator.characters_positions_in_vocabulary)
                    feed_dict = {self._pupil_hooks['validation_inputs']: vec}
                    feed_dict.update(additional_feed_dict)
                    fuse_operations = self._handler.get_tensors('fuse', char_idx)
                    fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                    self._handler.process_results(char_idx, fuse_res, regime='fuse')
                self._handler.start_text_accumulation()
                if fuse['fuse_stop'] == 'limit':
                    for char_idx in range(len(fuse['text']), len(fuse['text']) + fuse['max_num_of_chars'] ):
                        vec = batch_generator.pred2vec(fuse_res[0])
                        feed_dict = {self._pupil_hooks['validation_inputs']: vec}
                        feed_dict.update(additional_feed_dict)
                        fuse_operations = self._handler.get_tensors('fuse', char_idx)
                        fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                        self._handler.process_results(char_idx, fuse_res, regime='fuse')
                if fuse['fuse_stop'] == 'new_line':
                    char = None
                    counter = 0
                    char_idx = len(fuse['text'])
                    while char != '\n' and counter < fuse['max_num_of_chars']:
                        vec = batch_generator.pred2vec(fuse_res[0])
                        feed_dict = {self._pupil_hooks['validation_inputs']: vec}
                        feed_dict.update(additional_feed_dict)
                        fuse_operations = self._handler.get_tensors('fuse', char_idx)
                        fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                        self._handler.process_results(char_idx, fuse_res, regime='fuse')
                        char = batch_generator.vec2char(fuse_res[0], batch_generator.vocabulary)
                        counter += 1
                        char_idx += 1
                self._handler.stop_text_accumulation()
            self._handler.set_processed_fuse_index(None)
        self._handler.dispense_fuse_results(training_step)

    def _validate(self,
                  batch_generator_class,
                  validation_dataset,
                  validation_batch_size,
                  valid_batch_kwargs,
                  training_step=None,
                  additional_feed_dict=None,
                  save_to_file=None,
                  save_to_storage=None,
                  print_results=None):
        # print('valid_batch_kwargs:', valid_batch_kwargs)
        if 'reset_validation_state' in self._pupil_hooks:
            self._session.run(self._pupil_hooks['reset_validation_state'])
        #print('batch_generator_class:', batch_generator_class)
        valid_batches = batch_generator_class(validation_dataset[0], validation_batch_size, **valid_batch_kwargs)
        length = valid_batches.get_dataset_length()
        inputs, labels = valid_batches.next()
        step = 0
        self._handler.start_accumulation(validation_dataset[1], training_step=training_step)
        while step < length:
            validation_operations = self._handler.get_tensors('validation', step)
            feed_dict = {self._pupil_hooks['validation_inputs']: inputs,
                         self._pupil_hooks['validation_labels']: labels}
            if additional_feed_dict is not None:
                feed_dict.update(additional_feed_dict)
            valid_res = self._session.run(validation_operations, feed_dict=feed_dict)
            self._handler.process_results(training_step, valid_res, regime='validation')
            step += 1
            inputs, labels = valid_batches.next()

        means = self._handler.stop_accumulation(save_to_file=save_to_file,
                                                save_to_storage=save_to_storage,
                                                print_results=print_results)
        return means

    def _from_random_fuse(self):
        pass

    def _from_fuses(self):
        pass

    def _on_replicas(self):
        pass

    def _get_all_tensors_from_schedule(self, schedule):
        returned_list = list()
        for _, dict_with_tensors in schedule.items():
            for tensor_alias in dict_with_tensors.keys():
                returned_list.append(tensor_alias)
        return returned_list

    def _check_if_validation_is_needed(self, run_specs_set):
        """This method is not finished yet. Fuses, random and replicas should also be taken in account"""
        validation_is_needed = False
        for run_specs in run_specs_set:
            validation_is_needed = validation_is_needed or \
                                   not run_specs['train_specs']['no_validation']
        return validation_is_needed

    def _all_tensor_aliases_from_train_method_arguments(self, args_for_launches, evaluation=None):
        start_specs_for_launches, run_specs_for_launches = zip(*args_for_launches)
        list_of_required_tensors_aliases = list()
        result_types_for_launches = list()
        for start_specs in start_specs_for_launches:
            result_types_for_launches = add_missing_to_list(result_types_for_launches, start_specs['result_types'])
        list_of_required_tensors_aliases.extend(result_types_for_launches)
        for run_specs_set in run_specs_for_launches:
            if self._check_if_validation_is_needed(run_specs_set):
                for result_type in start_specs['result_types']:
                    list_of_required_tensors_aliases.append('validation_' + result_type)
        for run_specs_set in run_specs_for_launches:
            for run_specs in run_specs_set:
                train_aliases = self._get_all_tensors_from_schedule(run_specs['schedule']['train_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(list_of_required_tensors_aliases, train_aliases)
                valid_aliases = self._get_all_tensors_from_schedule(run_specs['schedule']['validation_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(list_of_required_tensors_aliases, valid_aliases)
        if evaluation is not None:
            if 'train' in evaluation['datasets'] and len(evaluation['datasets']) > 1:
                for result_type in evaluation['result_types']:
                    alias = 'validation_' + result_type
                    if alias not in list_of_required_tensors_aliases:
                        list_of_required_tensors_aliases.append(alias)
        return list_of_required_tensors_aliases

    @staticmethod
    def _tensor_aliases_from_schedule(schedule):
        tensor_aliases = list()
        for _, schedule in schedule.items():
            aliases = list(schedule.keys())
            tensor_aliases = add_missing_to_list(tensor_aliases, aliases)
        return tensor_aliases

    def _all_tensor_aliases_from_test_method_arguments(self, args):
        start_specs = args['start_specs']
        work = args['work']
        list_of_required_tensors_aliases = list()
        for res_type in start_specs['result_types']:
            list_of_required_tensors_aliases.append('validation_' + res_type)
        list_of_required_tensors_aliases = add_missing_to_list(
            list_of_required_tensors_aliases,
            self._tensor_aliases_from_schedule(work['fuse_tensors']))
        list_of_required_tensors_aliases = add_missing_to_list(
            list_of_required_tensors_aliases,
            self._tensor_aliases_from_schedule(work['validation_tensor_schedule']))
        return list_of_required_tensors_aliases

    def _create_missing_hooks(self, list_of_tensor_aliases, model_type='pupil'):
        missing = list()
        for tensor_alias in list_of_tensor_aliases:
            if model_type == 'pupil':
                if tensor_alias not in self._pupil_hooks:
                    missing.append(tensor_alias)
            if model_type == 'assistant':
                if tensor_alias not in self._assistant_hooks:
                    missing.append(tensor_alias)
        self.add_hooks(missing, model_type=model_type)

    def _build_batch_kwargs(self, unprepaired_kwargs):
        kwargs = dict()
        for key, arg in unprepaired_kwargs.items():
            if isinstance(arg, Controller):
                kwargs[key] = arg.get()
            else:
                kwargs[key] = arg
        return kwargs

    def _train(self,
               run_specs,
               checkpoints_path,
               batch_generator_class,
               init_step=0):
        """It is a method that does actual training and responsible for one training pass through dataset. It is called
        from train method (maybe several times)
        Args:
            kwargs should include all entries defined in self._pupil_default_training"""
        train_specs = construct(run_specs['train_specs'])
        schedule = construct(run_specs['schedule'])
        step = init_step

        # creating batch generator

        # resetting step in control_storage
        self.set_in_storage(step=step)
        learning_rate_controller = Controller(self._storage,
                                              train_specs['learning_rate'])
        train_feed_dict_additions = train_specs['additions_to_feed_dict']
        if train_feed_dict_additions is not None:
            additional_controllers = list()
            for addition in train_feed_dict_additions:
                additional_controllers.append(Controller(self._storage, addition))
        else:
            additional_controllers = None

        if train_specs['stop']['type'] == 'limit_steps':
            train_specs['stop']['limit'] += init_step
        should_continue = Controller(self._storage, train_specs['stop'])

        to_be_collected_while_training = schedule['to_be_collected_while_training']
        collect_interval = to_be_collected_while_training['results_collect_interval']
        print_per_collected = to_be_collected_while_training['print_per_collected']
        example_per_print = to_be_collected_while_training['example_per_print']

        if train_specs['no_validation'] or collect_interval is None:
            it_is_time_for_validation = Controller(self._storage,
                                                   {'type': 'always_false'})
            it_is_time_for_example = Controller(self._storage,
                                                {'type': 'always_false'})
        else:
            valid_period = collect_interval * print_per_collected
            it_is_time_for_validation = Controller(self._storage,
                                                   {'type': 'periodic_truth',
                                                    'period': valid_period})
            if example_per_print is None:
                it_is_time_for_example = Controller(self._storage,
                                                    {'type': 'always_false'})
            else:
                example_period = valid_period * example_per_print
                it_is_time_for_example = Controller(self._storage,
                                                    {'type': 'periodic_truth',
                                                     'period': example_period})

        if train_specs['checkpoint_steps'] is not None and checkpoints_path is not None:
            if train_specs['checkpoint_steps']['type'] == 'true_on_steps':
                for idx in range(len(train_specs['checkpoint_steps']['steps'])):
                    train_specs['checkpoint_steps']['steps'][idx] += init_step
            it_is_time_to_create_checkpoint = Controller(self._storage, train_specs['checkpoint_steps'])
        else:
            it_is_time_to_create_checkpoint = Controller(self._storage,
                                                         {'type': 'always_false'})

        batch_size_controller = Controller(self._storage, train_specs['batch_size'])
        batch_size_change_tracker_specs = Controller.create_change_tracking_specifications(train_specs['batch_size'])
        batch_size_should_change = Controller(self._storage, batch_size_change_tracker_specs)

        if train_specs['debug'] is not None:
            should_start_debugging = Controller(self._storage, train_specs['debug'])
        else:
            should_start_debugging = Controller(self._storage,
                                                {'type': 'true_on_steps',
                                                 'steps': []})

        train_batch_kwargs = dict()
        train_batch_kwargs_controller_specs = list()
        for key, batch_arg in train_specs['train_batch_kwargs'].items():
            if isinstance(batch_arg, dict):
                if 'type' in batch_arg:
                    train_batch_kwargs[key] = Controller(self._storage, batch_arg)
                    train_batch_kwargs_controller_specs.append(batch_arg)
                else:
                    train_batch_kwargs[key] = batch_arg
            else:
                train_batch_kwargs[key] = batch_arg
        change_tracker_specs = Controller.create_change_tracking_specifications(
            train_batch_kwargs_controller_specs)
        batch_generator_specs_should_change = Controller(self._storage, change_tracker_specs)

        controllers_for_printing = [learning_rate_controller]
        if additional_controllers is not None:
            controllers_for_printing.extend(additional_controllers)
        controllers_for_printing.append(batch_size_controller)
        batch_kwargs_controllers = list()
        for batch_kwarg in train_batch_kwargs.values():
            if isinstance(batch_kwarg, Controller):
                batch_kwargs_controllers.append(batch_kwarg)
        controllers_for_printing.extend(batch_kwargs_controllers)
        self._handler.set_new_run_schedule(schedule,
                                           train_specs['train_dataset'][1],
                                           [dataset[1] for dataset in train_specs['validation_datasets']])
        self._handler.set_controllers(controllers_for_printing)

        batch_size = batch_size_controller.get()
        tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
        train_batches = batch_generator_class(train_specs['train_dataset'][0], batch_size, **tb_kwargs)
        feed_dict = dict()
        while should_continue.get():
            if should_start_debugging.get():
                self._session = tf_debug.LocalCLIDebugWrapperSession(self._session)
                self._session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            if batch_size_should_change.get():
                batch_size = batch_size_controller.get()
                train_batches.change_batch_size(batch_size)

            if batch_generator_specs_should_change.get():
                tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
                train_batches.change_specs(**tb_kwargs)

            if it_is_time_to_create_checkpoint.get():
                self._create_checkpoint(step, checkpoints_path)

            learning_rate = learning_rate_controller.get()
            train_inputs, train_labels = train_batches.next()
            feed_dict[self._pupil_hooks['learning_rate']] = learning_rate
            if isinstance(self._pupil_hooks['inputs'], list):
                for input_tensor, input_value in zip(self._pupil_hooks['inputs'], train_inputs):
                    feed_dict[input_tensor] = input_value
            else:
                feed_dict[self._pupil_hooks['inputs']] = train_inputs
            if isinstance(self._pupil_hooks['labels'], list):
                for label_tensor, label_value in zip(self._pupil_hooks['labels'], train_labels):
                    feed_dict[label_tensor] = label_value
            else:
                feed_dict[self._pupil_hooks['labels']] = train_labels
            if train_feed_dict_additions is not None:
                for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
                    feed_dict[self._pupil_hooks[addition['name']]] = add_controller.get()
            train_operations = self._handler.get_tensors('train', step)
            #print('train_operations:', train_operations)
            #print('feed_dict:', feed_dict)
            train_res = self._session.run(train_operations, feed_dict=feed_dict)
            # here loss is given in bits per input (BPI)

            self._handler.process_results(step, train_res, regime='train')
            if it_is_time_for_validation.get():
                for validation_dataset in train_specs['validation_datasets']:
                    if train_feed_dict_additions is None:
                        valid_add_feed_dict = None
                    else:
                        valid_add_feed_dict = dict()
                        for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
                            valid_add_feed_dict[self._pupil_hooks[addition['placeholder']]] = add_controller.get()
                    _ = self._validate(batch_generator_class,
                                       validation_dataset,
                                       train_specs['validation_batch_size'],
                                       train_specs['valid_batch_kwargs'],
                                       training_step=step,
                                       additional_feed_dict=valid_add_feed_dict)
            if it_is_time_for_example.get():
                valid_add_feed_dict = dict()
                if train_feed_dict_additions is not None:
                    for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
                        valid_add_feed_dict[self._pupil_hooks[addition['placeholder']]] = add_controller.get()
                if schedule['fuses'] is not None:
                    self._on_fuses(train_batches,
                                   schedule['fuses'],
                                   training_step=step,
                                   additional_feed_dict=valid_add_feed_dict)
            step += 1
            self.set_in_storage(step=step)
        return step

    @staticmethod
    def _set_controller_name_in_specs(controller_specs, name):
        if isinstance(controller_specs, dict):
            if 'name' not in controller_specs:
                controller_specs['name'] = name

    def _process_abbreviation_in_1_entry(self, key, value):
        new_value = construct(value)
        if key == 'stop':
            if isinstance(value, int):
                new_value = {'type': 'limit_steps', 'limit': value}
            self._set_controller_name_in_specs(new_value, 'stop')
        if key == 'batch_size':
            if isinstance(value, int):
                new_value = {'type': 'fixed', 'value': value}
            self._set_controller_name_in_specs(new_value, 'batch_size')
        if key == 'num_unrollings':
            if isinstance(value, int):
                new_value = {'type': 'fixed', 'value': value}
            self._set_controller_name_in_specs(new_value, 'num_unrollings')
        if key == 'checkpoint_steps':
            if isinstance(value, list):
                new_value = {'type': 'true_on_steps', 'steps': value}
            elif isinstance(value, int):
                new_value = {'type': 'periodic_truth', 'steps': value}
            else:
                new_value = {'type': 'always_false'}
            self._set_controller_name_in_specs(new_value, 'checkpoint_steps')
        if key == 'learning_rate':
            self._set_controller_name_in_specs(new_value, 'learning_rate')
        if key == 'debug':
            if isinstance(value, int):
                new_value = {'type': 'true_on_steps', 'steps': [value]}
            else:
                new_value = None
            self._set_controller_name_in_specs(new_value, 'debug')
        return new_value

    def _process_abbreviations(self, set_of_kwargs, method_name):
        for key, value in set_of_kwargs.items():
            value = self._process_abbreviation_in_1_entry(key, value)
            set_of_kwargs[key] = value
        if search_in_nested_dictionary(set_of_kwargs, 'summary') is None:
            add_graph_to_summary = search_in_nested_dictionary(set_of_kwargs, 'add_graph_to_summary')
            train_summary_tensors = search_in_nested_dictionary(set_of_kwargs, 'train_summary_tensors')
            if train_summary_tensors is not None:
                if len(train_summary_tensors) > 0:
                    summary_tensors_provided = True
                else:
                    summary_tensors_provided = False
            else:
                summary_tensors_provided = False
            if add_graph_to_summary or summary_tensors_provided:
                set_of_kwargs['summary'] = True
            else:
                set_of_kwargs['summary'] = False
        self._process_datasets_shortcuts(set_of_kwargs)
        self._process_batch_kwargs_shortcuts(set_of_kwargs, method_name)

    def _process_abbreviations_in_hyperparameters_set(self, set):
        for key, values in set.items():
            new_values = list()
            for value in values:
                new_value = self._process_abbreviation_in_1_entry(key, value)
                new_values.append(new_value)
            set[key] = new_values
        return set

    def _process_batch_kwargs_shortcuts(self, set_of_kwargs, method_name):
        if method_name == 'train':
            if 'train_batch_kwargs' not in set_of_kwargs:
                set_of_kwargs['train_batch_kwargs'] = dict()
                if 'num_unrollings' in set_of_kwargs:
                    set_of_kwargs['train_batch_kwargs']['num_unrollings'] = set_of_kwargs['num_unrollings']
                    del set_of_kwargs['num_unrollings']
                if 'vocabulary' in set_of_kwargs:
                    set_of_kwargs['train_batch_kwargs']['vocabulary'] = set_of_kwargs['vocabulary']
                    del set_of_kwargs['vocabulary']
            if 'valid_batch_kwargs' not in set_of_kwargs:
                set_of_kwargs['valid_batch_kwargs'] = dict()
                if 'num_unrollings' in set_of_kwargs['train_batch_kwargs']:
                    set_of_kwargs['valid_batch_kwargs'] = {'num_unrollings': 1}
                if 'vocabulary' in set_of_kwargs['train_batch_kwargs']:
                    set_of_kwargs['valid_batch_kwargs']['vocabulary'] = list(
                        set_of_kwargs['train_batch_kwargs']['vocabulary'])
        if method_name == 'test':
            if 'valid_batch_kwargs' not in set_of_kwargs:
                set_of_kwargs['valid_batch_kwargs'] = dict()
                if 'num_unrollings' in set_of_kwargs:
                    set_of_kwargs['valid_batch_kwargs']['num_unrollings'] = {'num_unrollings': 1}
                    del set_of_kwargs['num_unrollings']
                if 'vocabulary' in set_of_kwargs:
                    set_of_kwargs['valid_batch_kwargs']['vocabulary'] = list(set_of_kwargs['vocabulary'])
                    del set_of_kwargs['vocabulary']

    def _process_datasets_shortcuts(self,
                                    set_of_kwargs):
        taken_names = list(self._datasets.keys())
        train_dataset = self._process_train_dataset_shortcuts(set_of_kwargs, taken_names)
        keys_to_remove = ['train_dataset', 'train_dataset_name', 'train_dataset_text', 'train_dataset_filename']
        for key in keys_to_remove:
            if key in set_of_kwargs:
                del set_of_kwargs[key]
        set_of_kwargs['train_dataset'] = train_dataset
        validation_datasets = self._process_validation_datasets_shortcuts(set_of_kwargs, taken_names)
        keys_to_remove = ['validation_datasets', 'validation_dataset_names',
                          'validation_dataset_texts', 'validation_dataset_filenames']
        for key in keys_to_remove:
            if key in set_of_kwargs:
                del set_of_kwargs[key]
        set_of_kwargs['validation_datasets'] = validation_datasets

    def _process_validation_datasets_shortcuts(self,
                                               set_of_kwargs,
                                               taken_names):
        validation_datasets = list()
        if 'validation_datasets' in set_of_kwargs:
            taken_names.extend(set_of_kwargs['validation_datasets'].keys())
            validation_datasets.extend(set_of_kwargs['validation_datasets'])
        if 'validation_dataset_names' in set_of_kwargs:
            for name in set_of_kwargs['validation_dataset_names']:
                if name not in self._datasets.keys():
                    raise InvalidArgumentError("Wrong value '%s' of variable '%s'\nAllowed values: '%s'" %
                                               (name,
                                                "set_of_kwargs['validation_dataset_names']",
                                                list(self._datasets.keys())))
                validation_datasets.append([self._datasets[name], name])
        if 'validation_dataset_texts' in set_of_kwargs:
            for text in set_of_kwargs['validation_dataset_texts']:
                key, value = self._process_input_text_dataset(text, taken_names)
                taken_names.append(key)
                validation_datasets.append([value, key])
        if 'validation_dataset_filenames' in set_of_kwargs:
            for filename in set_of_kwargs['validation_dataset_filenames']:
                key, value = self._process_dataset_filename(filename)
                taken_names.append(key)
                validation_datasets.append([value, key])
        return validation_datasets

    def _process_train_dataset_shortcuts(self,
                                         set_of_kwargs,
                                         taken_names):
        if 'train_dataset' in set_of_kwargs:
            taken_names.extend(set_of_kwargs['train_dataset'].keys())
            return set_of_kwargs['train_dataset']
        if 'train_dataset_name' in set_of_kwargs:
            if set_of_kwargs['train_dataset_name'] not in self._datasets.keys():
                raise InvalidArgumentError("Wrong value '%s' of variable '%s'\nAllowed values: '%s'" %
                                           (set_of_kwargs['train_dataset_name'], "set_of_kwargs['train_dataset_name']",
                                            list(self._datasets.keys())))
            return [self._datasets[set_of_kwargs['train_dataset_name']], set_of_kwargs['train_dataset_name']]
        if 'train_dataset_text' in set_of_kwargs:
            key, value =  self._process_input_text_dataset(set_of_kwargs['train_dataset_text'], taken_names)
            taken_names.append(key)
            return [value, key]
        if 'train_dataset_filename' in set_of_kwargs:
            key, value = self._process_dataset_filename(set_of_kwargs['train_dataset_filename'])
            taken_names.append(key)
            return [value, key]

    def _process_input_text_dataset(self, input, taken_names):
        idx = 0
        base = 'default_'
        new_key = base + str(idx)
        while new_key in taken_names:
            idx += 1
            new_key = base + str(idx)
        return new_key, input

    def _process_dataset_filename(self, input):
        splitted = input.split('/')
        self._datasets[splitted[-1]] = input
        return splitted[-1], input

    def _parse_1_set_of_kwargs(self,
                               kwargs_to_parse,
                               method_name,
                               repeated_key,
                               only_repeated,
                               old_arguments=None):
        # print('\n\n_parse_1_set_of_kwargs method:\nkwargs_to_parse:\n', kwargs_to_parse, '\nmethod_name:\n',
        #       method_name, '\nrepeated_key:\n', repeated_key, '\nonly_repeated:\n', only_repeated, '\nold_arguments:\n',
        #       old_arguments)
        kwargs_to_parse = construct(kwargs_to_parse)
        self._process_abbreviations(kwargs_to_parse, method_name)
        if old_arguments is None:
            if only_repeated:
                tmp = self.get_default_method_parameters(method_name)
                current_arguments = tmp[repeated_key]
            else:
                current_arguments = self.get_default_method_parameters(method_name)
        else:
            current_arguments = construct(old_arguments)

        for key, value in kwargs_to_parse.items():
            paste_into_nested_structure(current_arguments, key, value)

        #print('current_arguments:\n', current_arguments)
        return current_arguments

    def _parse_list_of_sets_of_kwargs(self,
                                      list_of_sets,
                                      method_name,
                                      repeated_key):
        # print('\n\n_parse_list_of_sets_of_kwargs method\nlist_of_sets:\n', list_of_sets, '\nmethod_name:\n', method_name,
        #       '\nrepeated_key:\n', repeated_key)
        parsed = self._parse_1_set_of_kwargs(list_of_sets[0],
                                             method_name,
                                             repeated_key,
                                             False)

        parsed[repeated_key] = [parsed[repeated_key]]

        repeated_parsed = parsed[repeated_key][0]
        for kwargs_set in list_of_sets[1:]:
            repeated_parsed = self._parse_1_set_of_kwargs(kwargs_set,
                                                          method_name,
                                                          repeated_key,
                                                          True,
                                                          old_arguments=repeated_parsed)
            parsed[repeated_key].append(repeated_parsed)
        #print('parsed:\n', parsed)
        return parsed

    def _parse_train_method_arguments(self,
                                      train_args,
                                      train_kwargs,
                                      set_passed_parameters_as_default=False):
        """Performs parsing of 'train' and 'train_assistant' method arguments. Optionally updates
        self._pupil_default_training or self._assistant_default_training.
        Args:
            train_args: args passed to train method
            set_passed_parameters_as_default: defines if reset of default parameters is needed
            train_kwargs: kwargs passed to train method
        Returns:
            a dictionary of start parameters (same format as self._default_start)
            a list of dictionaries with all parameters required for launch (each dictionary has the same format as
                self._pupil_default_training or self._assistant_default_training)"""

        #print('\n\n_parse_train_method_arguments method\ntrain_args:', train_args, '\ntrain_kwargs:', train_kwargs)
        if len(train_args) == 0:
            #print('train_kwargs:', train_kwargs)
            parsed_arguments = self._parse_list_of_sets_of_kwargs([train_kwargs],
                                                                  'train',
                                                                  'run')
        else:
            parsed_arguments = self._parse_list_of_sets_of_kwargs(train_args,
                                                                  'train',
                                                                  'run')

        return parsed_arguments

    def train(self,
              *args,
              start_session=True,
              close_session=True,
              set_passed_parameters_as_default=False,
              **kwargs):
        """The method responsible for model training. User may specify what intermediate results he wishes to
        collect. He may regulate learning process (see arguments). It is also possible to start learning from a check
        point. User may choose if he wishes to limit number of steps
        Args:
            args: A list of arbitrary number of dictionaries which entries are similar to structure of kwargs. It is
                used if user wishes to train model consequently on several datasets. If any dictionary in list contains
                less entries than the previous one, missing entries are taken from previous. If the first doesn't have
                all entries missing entries are filled with default values
            start_session: shows if new session should be created or already opened should be used
            close_session: shows if session should be closed at the end of training
            set_passed_parameters_as_default: if True parameters of launch are saved to self._pupil_default_training.
                If args are provided the first args[0] is used for self._pupil_default_training resetting
            kwargs:
                This argument specifies the learning should be performed. There are many options and it is not
                necessary to provide all kwargs - missing will be filled with default values specified in
                _default_train_method_args atribute
                allow_soft_placement: if True tensorflow is allowed to override device assignments specified by user and
                    put ops on available devices
                gpu_memory: memory fraction tensorflow allowed to allocate. If None all available memory is allocated
                log_device_placement: If True device placements are printed to console
                restore_path: If provided graph will be restored from checkpoint
                save_path: path to directory where all results are saved
                result_types: specifies what types of results should be collected. loss, perplexity, accuracy, bpc are
                    available
                summary: If True summary writing is activated
                add_graph_to_summary: If True graph is added to summary
                batch_generator_class: class of batch generator. It has to have certain methods for correct functioning
                assistant: If meta learning is used for model training it is name of assistant network
                learning_rate: specifications for learning_rate control. If it is a float learning rate will not change
                    while learning. Otherwise it should be a dictionary. Now only exponential decay option is availbale.
                    Below dictionary entries are described
                    exponential decay:
                        type: str 'exponential_decay'
                        init: float, initial learning rate
                        decay: a factor on which learning rate is multiplied every period of steps
                        period: number of steps after which learning rate is being decreased
                additions_to_feed_dict: If your model requires some special placeholders filling (e. g. probability
                    distribution for a stochastic node) it is provided through additions_to_feed_dict. It is a
                    dictionary which keys are tensor aliases in _pupil_hooks attribute and values are dictionaries
                    of the same structure as learning_rate
                stop: specifies when learning should be stopped. It is either an integer (number of steps after which
                    learning is being stopped) or a dictionary of the same structure as learning_rate where you may
                    specify custom way of learning interruption
                train_dataset: A dataset on which model will be trained. It can be a name of dataset provided earlier to
                    Environment constructor or just something what you wish to pass to batch generator (file name, str,
                    etc.)
                batch_size: integer or dictionary of the same type as learning_rate if you wish to somehow change batch
                    size during learning
                train_batch_kwargs: If your batch generator requires some specific arguments they can be provided
                    through this dictionary (for example num_unrollings). This dictionary is used for batch generator
                    construction for training (any of batch generator parameters can be provided as key word args
                    separately if their processing is described in _process_batch_kwargs_shortcut method. Now it is only
                    'vocabulary' and 'num_unrollings')
                checkpoint_steps: list of steps on which checpoints should be created
                debug: step on which tfdbg should be activated. Default is None
                validation_dataset_names: list of dataset names used for validation (datasets have to provided to
                    Environment instance separately. Now only through constructor
                validation_dataset_texts: list of texts (type str) used for validation
                validation_dataset_filenames: file names of datasets used for validation
                  (if validation_dataset_names, validation_dataset_texts, validation_dataset_filenames provided together
                   all of them are used)
                validation_batch_size: batch size for validation
                valid_batch_kwargs: same as train_batch_kwargs
                to_be_collected_while_training: a dictionary with 3 entries (all of them can be provided independently)
                    results_collect_interval: number of steps after which data is collected
                    print_per_collected: every print_per_collected-th point collected with results_collect_interval
                        schedule is printed
                    example_per_print: every example_per_print print examples of model functioning are printed
                        (continuing from random letter, from specified fuse, responding on user specified replicas)
                printed_result_types: what model should print. Default is loss. perplexity, accuracy, bpc are also
                    available
                printed_controllers: if during learning some hyperparameters are changing you may print them to
                    console. Default printed is learning rate
                fuses: specifies fuses from which model should periodically generate text. This option is not
                    available yet
                fuse_tensors: tensor aliases from _pupil_hooks attribute which should be either saved or printed.
                    not available
                replicas: If dialog agent is trained it can be tested with consequently feeding it with few user
                    specified replicas. It can be used to check if agent is capable of dialog context accumulating
                random: NLP agents can be tested on text generating task. It is provided with first character and
                    then tries to generate text. This argument is responsible for specifying how many times it will
                    be performed and specifying length of generated sequences (not available)
                train_tensor_schedule: If user wishes he may print or save any tensor in the graph (not available)
                valid_tensor_schedule: same as train_tensor_schedule"""
        tmp_output = self._parse_train_method_arguments(args,
                                                        kwargs,
                                                        set_passed_parameters_as_default=
                                                        set_passed_parameters_as_default)
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        run_specs_set = tmp_output['run']
        all_tensor_aliases = self._all_tensor_aliases_from_train_method_arguments([(start_specs, run_specs_set)])
        self._create_missing_hooks(all_tensor_aliases)

        if start_session:
            self._start_session(session_specs['allow_soft_placement'],
                                session_specs['log_device_placement'],
                                session_specs['gpu_memory'],
                                session_specs['visible_device_list'])
        self._train_repeatedly(start_specs, run_specs_set)
        if close_session:
            self._close_session()

    def _train_repeatedly(self, start_specs, run_specs_set):
        # initializing model
        self.flush_storage()
        self._initialize_pupil(start_specs['restore_path'])

        # print('start_specs:', start_specs)

        self._handler = Handler(self,
                                self._pupil_hooks,
                                'train',
                                start_specs['save_path'],
                                start_specs['result_types'],
                                summary=start_specs['summary'],
                                add_graph_to_summary=start_specs['add_graph_to_summary'],
                                batch_generator_class=start_specs['batch_generator_class'],
                                vocabulary=start_specs['vocabulary'])
        if start_specs['save_path'] is not None:
            checkpoints_path = start_specs['save_path'] + '/checkpoints'
            create_path(checkpoints_path)
        else:
            checkpoints_path = None
        init_step = 0
        for run_specs in run_specs_set:
            init_step = self._train(run_specs,
                                    checkpoints_path,
                                    start_specs['batch_generator_class'],
                                    init_step=init_step)
        if checkpoints_path is not None:
            self._create_checkpoint('final', checkpoints_path)
        self._handler.close()

    def _several_launches_without_rebuilding(self,
                                             queue,
                                             kwargs_for_building,
                                             session_specs,
                                             args_for_launches,
                                             evaluation):

        self._build(kwargs_for_building)
        #print('args_for_launches:', args_for_launches)
        all_tensor_aliases = self._all_tensor_aliases_from_train_method_arguments(args_for_launches, evaluation=evaluation)
        #print('all_tensor_aliases:', all_tensor_aliases)
        self._create_missing_hooks(all_tensor_aliases)
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['visible_device_list'])
        datasets = dict(evaluation['datasets'].items())
        if 'train' in datasets:
            del datasets['train']
        if evaluation['batch_gen_class'] is None:
            eval_batch_gen_class = self._default_batch_generator
        else:
            eval_batch_gen_class = evaluation['batch_gen_class']
        for start_specs, run_specs_set in args_for_launches:
            result = dict()
            self._train_repeatedly(start_specs, run_specs_set)
            if 'train' in evaluation['datasets']:
                tr_res = dict()
                for key, res in self._storage['train'].items():
                    if len(res) > 0:
                        tr_res[key] = res[-1]
                result['train'] = tr_res
            self._handler = Handler(self,
                                    self._pupil_hooks,
                                    'test',
                                    None,
                                    evaluation['result_types'])
            for dataset_name, dataset in datasets.items():
                #print('dataset_name:', dataset_name)
                #print('dataset:', dataset)
                means = self._validate(eval_batch_gen_class,
                                       dataset,
                                       evaluation['batch_size'],
                                       evaluation['batch_kwargs'],
                                       additional_feed_dict=evaluation['additional_feed_dict'],
                                       save_to_file=False,
                                       save_to_storage=False,
                                       print_results=False)
                result[dataset_name] = means
            #print('result in process:', result)
            queue.put(result)

    def _form_list_of_kwargs(self, kwargs, hyperparameters):
        output = [(construct(kwargs), dict(), list())]
        lengths = list()
        for name, values in hyperparameters.items():
            new_output = list()
            lengths.append(len(values))
            for base in output:
                for idx, value in enumerate(values):
                    new_base = construct(base)
                    # print('in _form_list_of_kwargs:')
                    # print('value:', value)
                    paste_into_nested_structure(new_base[0], name, value)
                    print("new_base[0]['run'][0]['train_specs']['learning_rate']:",
                          new_base[0]['run'][0]['train_specs']['learning_rate'])
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

    def several_launches(self,
                         evaluation,
                         kwargs_for_building,
                         args_for_launches=None,
                         build_hyperparameters=dict(),
                         other_hyperparameters=dict(),
                         **kwargs):
        tmp_output = self._parse_train_method_arguments([],
                                                        kwargs,
                                                        set_passed_parameters_as_default=False)
        session_specs = tmp_output['session_specs']
        build_hyperparameters = self._process_abbreviations_in_hyperparameters_set(construct(build_hyperparameters))
        other_hyperparameters = self._process_abbreviations_in_hyperparameters_set(construct(other_hyperparameters))
        #print('build_hyperparameters:', build_hyperparameters)
        #print('other_hyperparameters:', other_hyperparameters)
        list_of_build_kwargs = self._pupil_class.form_list_of_kwargs(kwargs_for_building,
                                                                     build_hyperparameters)

        list_of_build_kwargs, list_of_build_hp_values, _ = zip(*list_of_build_kwargs)
        base = tmp_output
        del base['session_specs']
        args_for_launches = self._form_list_of_kwargs(base, other_hyperparameters)
        # print("args_for_launches[0][0]['start_specs']['save_path']:",
        #       args_for_launches[0][0]['start_specs']['save_path'])

        args_for_launches, hp_values_list, _ = zip(*args_for_launches)

        refactored = list()
        for args in args_for_launches:
            start_specs = args['start_specs']
            run_specs_set = args['run']
            refactored.append((start_specs, run_specs_set))
        args_for_launches = refactored
        hp = list(build_hyperparameters.keys())
        hp.extend(list(other_hyperparameters.keys()))
        self._handler = Handler(self,
                                self._pupil_hooks,
                                'several_launches',
                                evaluation['save_path'],
                                evaluation['result_types'],
                                eval_dataset_names=list(evaluation['datasets'].keys()),
                                hyperparameters=hp)
        for build_kwargs, build_hp_values in zip(list_of_build_kwargs, list_of_build_hp_values):
            queue = mp.Queue()
            p = mp.Process(target=self._several_launches_without_rebuilding,
                           args=(queue, build_kwargs, session_specs, args_for_launches, evaluation))
            p.start()
            for idx in range(len(args_for_launches)):
                hp = hp_values_list[idx]
                res = queue.get()
                #print('res:', res)
                hp.update(build_hp_values)
                self._handler.process_results(hp, res, regime='several_launches')
            p.join()

    def inference(self,
                  restore_path,
                  log_path,
                  vocabulary,
                  characters_positions_in_vocabulary,
                  batch_generator_class,
                  gpu_memory=None,
                  appending=True,
                  temperature=0.,
                  first_speaker='human'):
        create_path(log_path, file_name_is_in_path=True)
        if not appending:
            log_path = add_index_to_filename_if_needed(log_path)
        if appending:
            fd = open(log_path, 'a', encoding='utf-8')
        else:
            fd = open(log_path, 'w', encoding='utf-8')
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        if gpu_memory is not None:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory

        with tf.Session(config=config) as session:
            if restore_path is None:
                print_and_log('Skipping variables restoring. Continueing on current variables values', fd=fd)
            else:
                print('restoring from %s' % restore_path)
                self._pupil_hooks['saver'].restore(session, restore_path)
            self._pupil_hooks['reset_validation_state'].run()
            if first_speaker == 'human':
                human_replica = input('Human: ')
            else:
                human_replica = ''
            sample_prediction = self._pupil_hooks['validation_predictions']
            sample_input = self._pupil_hooks['validation_inputs']
            while not human_replica == 'FINISH':
                if human_replica != '':
                    print_and_log('Human: ' + human_replica, _print=False, fd=fd)
                    for char in human_replica:
                        feed = batch_generator_class.char2vec(char, characters_positions_in_vocabulary)
                        # print('feed.shape:', feed.shape)
                        _ = sample_prediction.eval({sample_input: feed})
                feed = batch_generator_class.char2vec('\n', characters_positions_in_vocabulary)
                prediction = sample_prediction.eval({sample_input: feed})
                if temperature != 0.:
                    prediction = apply_temperature(prediction, -1, temperature)
                    prediction = sample(prediction, -1)
                counter = 0
                char = None
                bot_replica = ""
                # print('ord(\'\\n\'):', ord('\n'))
                while char != '\n' and counter < 10:
                    feed = batch_generator_class.pred2vec(prediction)
                    # print('prediction after sampling:', prediction)
                    # print('feed:', feed)
                    prediction = sample_prediction.eval({sample_input: feed})
                    # print('prediction before sampling:', prediction)
                    if temperature != 0.:
                        prediction = apply_temperature(prediction, -1, temperature)
                        # print('prediction after temperature:', prediction)
                        prediction = sample(prediction, -1)
                    char = batch_generator_class.vec2char(np.reshape(feed, (1, -1)), vocabulary)[0]
                    if char != '\n':
                        # print('char != \'\\n\', counter = %s' % counter)
                        # print('ord(char):', ord(char))
                        bot_replica += char
                    counter += 1
                print_and_log('Bot: ' + bot_replica, fd=fd)
                feed = batch_generator_class.char2vec('\n', characters_positions_in_vocabulary)
                _ = sample_prediction.eval({sample_input: feed})

                human_replica = input('Human: ')
            fd.write('\n*********************')
            fd.close()