import codecs
import numpy as np
import random

categories = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']

dict_file = codecs.open('processed/dictionary.txt', 'r', 'utf_8_sig')

dictionary = []
for line in dict_file:
    line = line[: len(line) - 1]
    dictionary.append(line)


def similar_words(word1, word2, coef = .5):
    if len(word1) == len(word2):
        ch = 0
        n = len(word1)
        zn = 0
        for i in range(n):
            zn += np.sqrt(n-i)
        for i in range(n):
            if word1[i] == word2[i]:
                ch+=np.sqrt(n-i)
        if ch/zn >= coef:
            return True
        else:
            return False
    else:
        return False


def remove_punctuation(word):
    punctuation = ['!', ':', ':', ',', '.', '?', "'", '"', '(', ')', '«', '»', '+', '-', '=', '_', '/', '\\', '|', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    new_word = ''
    for symbol in word:
        if symbol not in punctuation:
            new_word += symbol
    return new_word.lower()


def line2vec(line, dictionary):
    vector = [0] * len(dictionary)

    for word in line.split():
        word = remove_punctuation(word)

        for d in dictionary:
            if similar_words(word, d):
                vector[dictionary.index(d)] += 1
    return vector


train_file = codecs.open('news_train.txt', 'r', 'utf_8_sig')

input_vectors = []
outputs = []
for line in train_file:
    label, name, content = line.split('\t')

    vector = line2vec(name, dictionary)
    output = [0]*10
    output[categories.index(label)] = 1

    input_vectors.append(vector)
    outputs.append(output)


train_vectors_i = codecs.open('processed/train_vectors_input.txt', 'w+', 'utf_8_sig')
train_vectors_o = codecs.open('processed/train_vectors_outputs.txt', 'w+', 'utf_8_sig')

for i in input_vectors:
    train_vectors_i.write(str(i) + '\n')

for i in outputs:
    train_vectors_o.write(str(i) +'\n')

print('text processed')