import tensorflow as tf
from environment import Environment
from simple_fontain import SimpleFontain, SimpleFontainBatcher
from some_useful_functions import load_vocabulary_from_file, get_positions_in_vocabulary

f = open('small_flagged_subs.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
offset = 10000
valid_size = 1000
valid_text = text[offset:offset + valid_size]
train_text = text[offset + valid_size:]
train_size = len(train_text)

vocabulary = load_vocabulary_from_file('subs_vocabulary.txt')

cpiv = get_positions_in_vocabulary(vocabulary)

vocabulary_size = len(vocabulary)
#print(vocabulary_size)

env = Environment(SimpleFontain, SimpleFontainBatcher)
env.build(batch_size=64,
          vocabulary_size=vocabulary_size,
          attention_interval=3,
          attention_visibility=5,
          subsequence_length_in_intervals=7,
          characters_positions_in_vocabulary=cpiv)

def count_non_zeros(**kwargs):
    tensor = kwargs['tensor']
    return tf.reduce_sum(tf.to_float(tf.not_equal(tensor, 0.)))

env.register_build_function(count_non_zeros, 'count_nz')

tmpl = "train/saved_state_layer%s_number%s:0"
schedule = dict()
for i in range(3):
    for j in range(2):
        hook_name = 'nz_ss_l%s_n%s' % (i, j)
        env.register_builder('count_nz',
                             tensor_names={'tensor': tmpl % (i, j)},
                             output_hook_name=hook_name)
        schedule[hook_name] = [i for i in range(30)]


env.train(save_path='debugging_simple_fontain/first',
          learning_rate={'type': 'exponential_decay',
                         'init': 5.,
                         'decay': .9,
                         'period': 500},
          batch_size=64,
          vocabulary=vocabulary,
          checkpoint_steps=[100],
          stop=1000,
          num_unrollings=21,
          #debug=0,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          printed_result_types=['loss', 'perplexity', 'accuracy'],
          #add_graph_to_summary=True,
          #validation_dataset=[valid_text],
          results_collect_interval=10,
          no_validation=True,
          train_print_tensors=schedule)