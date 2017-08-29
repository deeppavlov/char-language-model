# this script removes all wrong letters
import sys

input_filename = sys.argv[1]
output_filename = sys.argv[2]
length = int(sys.argv[3])

input_f = open(input_filename, 'r', encoding='utf-8')

text = input_f.read(length)


output_f = open(output_filename, 'w', encoding='utf-8')
output_f.write(text)

input_f.close()
output_f.close() 
