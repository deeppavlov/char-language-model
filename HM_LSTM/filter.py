# this script removes all wrong letters
import sys

lowercase = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
uppercase = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
other = " \n.,;:()?!\"-"


input_filename = sys.argv[1]
output_filename = sys.argv[2]
if len(sys.argv) > 3:
    allowed_characters_filename = sys.argv[3]
    with open(allowed_characters_filename, 'r', encoding='utf-8') as characters_f:
        lines = characters_f.readlines()
        lowercase = lines[0][:-1]
        uppercase = lines[1][:-1]
        not_processed = lines[2][:-1]
        other=""
        backslash_switch = False
        for idx, nchar in enumerate(not_processed):
            if backslash_switch:
                if nchar == 'n':
                    other += '\n'
                elif nchar == 't':
                    other += '\t'
                elif nchar == 'r':
                    other += '\r'
                elif nchar == '\\':
                    other += '\\'
                else:
                    other += '\\'
                backslash_switch = False
            else:
                if nchar == '\\':
                    backslash_switch = True
                else:
                    other += nchar 
 
                      


print('lowercase:', lowercase)
print('uppercase:', uppercase)
output_other = ""
for other_char in other:
    if other_char == '\n':
        output_other += '\\'
        output_other += 'n'
    elif other_char == '\t':
        output_other += '\\'
        output_other += 't'
    elif other_char == '\r':
        output_other += '\\'
        output_other += 'r'
    elif other_char == '\"':
        output_other += '\"'
    elif other_char == '\\':
        output_other += '\\'
    else:
        output_other += other_char
            
print('other:', output_other)

allowed_characters = list()
allowed_characters.extend(lowercase)
allowed_characters.extend(uppercase)
allowed_characters.extend(other)

input_f = open(input_filename, 'r', encoding='utf-8')
output_f = open(output_filename, 'w', encoding='utf-8')
text = input_f.read()
print('text length:', len(text))
new_text = ""
for char in text:
    if char in allowed_characters:
        new_text += char
print ('new text length:', len(new_text))
output_f.write(new_text)

input_f.close()
output_f.close() 
