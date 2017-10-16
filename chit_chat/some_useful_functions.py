import numpy as np
import inspect
import os


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


def id2char(dictid, vocabulary):
    voc_size = len(vocabulary)
    if (dictid >= 0) and (dictid < voc_size):
        return vocabulary[dictid]
    else:
        print(u"unexpected id")
        return u'\0'


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


def flatten(nested):
    if not isinstance(nested, (tuple, list)):
        return [nested]
    output = list()
    for inner_object in nested:
        flattened = flatten(inner_object)
        output.extend(flattened)
    return output