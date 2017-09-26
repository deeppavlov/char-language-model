
from environment import Environment
from vanilla import Vanilla, BatchGenerator
from vanilla import create_vocabulary

f = open('enwik8_clean', 'r', encoding='utf-8')
text = f.read()
f.close()

# different
offset = 10000
valid_size = 1000
valid_text = text[offset:offset + valid_size]
train_text = text[offset + valid_size:]
train_size = len(train_text)

# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Vanilla, BatchGenerator)
env.build(batch_size=64,
          num_nodes=250,
          vocabulary_size=vocabulary_size,
          num_unrollings=10)

env.train(save_path='debugging_environment/first',
          learning_rate={'type': 'exponential_decay',
                         'init': 0.5,
                         'decay': .9,
                         'period': 500},
          batch_size=64,
          num_unrollings=10,
          vocabulary=vocabulary,
          checkpoint_steps=[100],
          stop=1000,
          #train_dataset_text='abx',
          #validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          #validation_dataset=[valid_text],
          results_collect_interval=400)