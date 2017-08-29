# this script removes all wrong letters
import sys
import re

def get_line(f, delimeter='>'):
    c = f.read(1)
    #print(ord(c))
    line = c
    while c and (c != delimeter):
        c = f.read(1)
        if c:
            line += c
        #print(ord(c))
    return line 

lowercase = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
uppercase = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
letters = lowercase + uppercase
other = " \.\n,;:\(\)\"-"
filter_regex = '[' + '^' + lowercase + uppercase + other + ']'
letters_list = list()
letters_list.extend(lowercase)
letters_list.extend(uppercase)

input_filename = sys.argv[1]
output_filename = sys.argv[2]

input_f = open(input_filename, 'r', encoding='utf-8')
output_f = open(output_filename, 'w', encoding='utf-8')

good_line = False
table = False
header = False
sense = True

line = get_line(input_f)
#print(line)
while line:
    if '<content' in line:
        good_line = True
    if good_line:
        if '<table' in line:                                                          # remove all tables <table>...</table>
            table = True
        if '<h>' in line:                                                             # remove all headers <h>...</h>
            header = True
        if '</content' in line:
            good_line = False
        initial_line = str(line)
        line = re.sub('<.*>', '', line)                                               # remove xml tags
        line = re.sub('&amp;', '&', line)                                             # decode URL encoded chars
        line = re.sub('&lt;', '<', line)
        line = re.sub('&gt;', '>', line)    
        line = re.sub('<ref[^<]*<\/ref>', '', line)                                   # remove references <ref...> ... </ref>
        line = re.sub('<[^>]*>', '', line)                                            # remove xhtml tags
        line = re.sub('<\[http:[^] ]*', '[', line)                                    # remove normal url, preserve visible text
        line = re.sub('<\|thumb', '', line, flags=re.I)                               # remove images links, preserve caption
        line = re.sub('<\|left', '', line, flags=re.I)
        line = re.sub('<\|right', '', line, flags=re.I)
        line = re.sub('<\|\d+px', '', line, flags=re.I)
        line = re.sub('\[\[image:[^\[\]]*\|', '', line, flags=re.I)
        line = re.sub('\[\[category:([^|\]]*)[^]]*\]\]', '[[\1]]', line, flags=re.I)  # show categories without markup
        line = re.sub('\[\[[a-z\-]*:[^\]]*\]\]', '[', line)                           # remove links to other languages
        line = re.sub('\[\[[^\|\]]*\|', '[[', line)                                   # remove wiki url, preserve visible text
        line = re.sub('\{\{[^}]*}}', '', line)                                        # remove {{icons}} and {tables}
        line = re.sub('\{[^}]*}', '', line) 
        line = re.sub('\[', '', line)                                                 # remove [ and ]
        line = re.sub('\]', '', line)
        line = re.sub('&[^;]*;', '', line)                                            # remove URL encoded chars
        # spell digits
        line = re.sub('1', ' один ', line)
        line = re.sub('2', ' два ', line)
        line = re.sub('3', ' три ', line)
        line = re.sub('4', ' четыре ', line)
        line = re.sub('5', ' пять ', line)
        line = re.sub('6', ' шесть ', line)
        line = re.sub('7', ' семь ', line)
        line = re.sub('8', ' восемь ', line)
        line = re.sub('9', ' девять ', line)
        line = re.sub('0', ' ноль ', line)
        line = re.sub('\/', ',', line)
        line = re.sub('\'+', '\"', line)
        line = re.sub('=+', '\"', line)
        line = re.sub(filter_regex, '', line)
        line = re.sub('\"[ ]*\"', ' ', line)
        line = re.sub('\"+', '\"', line)
        line = re.sub('[\{\}]', '', line)
        line = re.sub('\([ ,\.;:]*\)', ' ', line)
        line = re.sub('[ ]*\n', '\n', line)
        line = re.sub('\n[ ]*', '\n', line)
        line = re.sub('\n{3,}', '\n\n', line)
        line = re.sub(' +', ' ', line)
        line = re.sub('\( ', '(', line)
        line = re.sub(' \)', ')', line)
        line = re.sub('[ \n]*\.', '.', line)
        line = re.sub('[ \n]*,', ',', line)
        line = re.sub(',+', ',', line)

        there_is_sense_in_line = False
        if not (table or header):
            for char in line:
                there_is_sense_in_line = there_is_sense_in_line or (char in letters)
        if not sense and there_is_sense_in_line:
            output_f.write('\n')
        sense = there_is_sense_in_line
        """if table:
            print(line)"""
        if (not table) and (not header) and sense:
            output_f.write(line)
        if '</table>' in initial_line:
            table = False
        if '</h>' in initial_line:
            header = False
    line = get_line(input_f)
    #print(line)



output_f.close()

