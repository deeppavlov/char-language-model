# this script removes all wrong letters
import sys
import re


lowercase = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
uppercase = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
other = " ,;:\(\)\"-"
allowed_characters_string = lowercase + uppercase + other
letters_list = list()
letters_list.extend(lowercase)
letters_list.extend(uppercase)

possible_edges = list(letters_list)
possible_edges.extend(['(', ')', '\"'])

input_filename = sys.argv[1]
output_filename = sys.argv[2]


def get_string_type(string):
    if string == '':
        return 0
    if string[0] in delimeters:
        return 2
    no_sense = True
    for char in string:
        no_sense = no_sense and (not (char in letters_list))
    if no_sense:
        return 0
    else: 
        return 1
    

def remove_leading_and_trailing(string):
    start = 0
    while not string[start] in possible_edges:
        start += 1
    end = -1
    while not string[end] in possible_edges:
        end -= 1
    if end  == -1:
        return string[start:]
    return string[start : end+1]

delimeters = ['.', '!', '?']
delimeters_for_regex = ['\\' + delimeter for delimeter in delimeters]
split_regex = ""
for delimeter in delimeters_for_regex:
    split_regex += delimeter
    split_regex += '|'
split_regex = '(' + split_regex[:-1] +')'

input_f = open(input_filename, 'r', encoding='utf-8')

text = input_f.read()
lines = re.split(split_regex, text)

new_text = ""

filter_regex = '[^' + allowed_characters_string + ']'
for i, line in enumerate(lines):
    if get_string_type(line) == 1:
        new_text += '\n'
        filtered = re.sub(filter_regex, ' ', line)
        filtered = re.sub(' +', ' ', filtered)
        filtered = remove_leading_and_trailing(filtered)
        new_text += filtered
    elif get_string_type(line) == 2:
        new_text += line
new_text = new_text[1:]

output_f = open(output_filename, 'w', encoding='utf-8')
output_f.write(new_text)

input_f.close()
output_f.close() 
