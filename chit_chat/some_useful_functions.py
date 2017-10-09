import numpy as np
import inspect


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


def char2id(char, characters_positions_in_vocabulary):
    if char in characters_positions_in_vocabulary:
        return characters_positions_in_vocabulary[char]
    else:
        print(u'Unexpected character: %s\nUnexpected character number: %s\n' % (char, ord(char)))
        return None


def filter_text(text, allowed_letters):
    new_text = ""
    for char in text:
        if char in allowed_letters:
            new_text += char
    return new_text


def char2vec(char, characters_positions_in_vocabulary):
    voc_size = len(characters_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float)
    vec[0, char2id(char, characters_positions_in_vocabulary)] = 1.0
    return vec


def create_and_save_vocabulary(input_file_name,
                               vocabulary_file_name):
    input_f = open(input_file_name, 'r', encoding='utf-8')
    text = input_f.read()
    output_f = open(vocabulary_file_name, 'w', encoding='utf-8')
    vocabulary = create_vocabulary(text)
    vocabulary_string = ''.join(vocabulary)
    output_f.write(vocabulary_string)
    input_f.close()
    output_f.close()


def load_vocabulary_from_file(vocabulary_file_name):
    input_f = open(vocabulary_file_name, 'r', encoding='utf-8')
    vocabulary_string = input_f.read()
    return list(vocabulary_string)


def construct(obj):
    """Used for preventing of not expected changing of class attributes"""
    if isinstance(obj, dict):
        new_obj = dict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif isinstance(obj, list):
        new_obj = list()
        for value in obj:
            new_obj.append(construct(value))
    elif isinstance(obj, tuple):
        base = list()
        for value in obj:
            base.append(construct(value))
        new_obj = tuple(base)
    elif isinstance(obj, str):
        new_obj = str(obj)
    elif isinstance(obj, (int, float, complex, type(None))) or inspect.isclass(obj):
        new_obj = obj
    else:
        raise TypeError("Object of unsupported type was passed to construct function: %s" % type(obj))
    return new_obj