import codecs

categories = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']

files = []
for cat in categories:
    files.append(codecs.open('processed/roots_for_{}.txt'.format(cat), 'r', 'utf_8_sig'))

dictionary_list = []

for i in range(10):
    temp_list = []
    for line in files[i]:
        line = line[: len(line) - 1]
        temp_list.append(line)
    dictionary_list += temp_list[: -64]

dictionary = {i: dictionary_list.count(i) for i in dictionary_list}

new_dict = {}

for i in dictionary:
    if dictionary[i] < 5:
        new_dict[i] = dictionary[i]
dictionary = new_dict


dict_file = codecs.open('processed/dictionary.txt', 'w+', 'utf_8_sig')
k = dictionary.keys()
for i in k:
    dict_file.write(i + '\n')        