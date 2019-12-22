import codecs
import numpy as np
import random
from copy import deepcopy

categories = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']

dict_file = codecs.open('processed/dictionary.txt', 'r', 'utf_8_sig')

dictionary = []
for line in dict_file:
    line = line[: len(line) - 1]
    dictionary.append(line)

dl = len(dictionary)
neuron_number = 6
weights = [[[0 for i in range(neuron_number)] for j in range(dl)], [[0 for i in range(dl)] for j in range(neuron_number)]]


train_vectors_i = codecs.open('processed/train_vectors_input.txt', 'r', 'utf_8_sig')
train_vectors_o = codecs.open('processed/train_vectors_outputs.txt', 'r', 'utf_8_sig')

input_vectors = []
outputs = []

for line in train_vectors_i:
    line2 = line[1:-2]
    input_vector = line2.split(', ')
    input_vectors.append([int(i) for i in input_vector])

for line in train_vectors_o:
    line2 = line[1:-2]
    output_vector = line2.split(', ')
    outputs.append([int(i) for i in output_vector])

print('read')

'''
dl = 2
neuron_number = 3
'''

def ReLU(x, coef = 1):
    if x >=0:
        return x * coef
    else:
        return 0


def classify(output_vector):
    return categories[output_vector.index(max(output_vector))]


def propogation(input_vector, weights, dictionary_length = dl, neuron_number = neuron_number, activation_function = ReLU):
    hidden_layer = [0] * neuron_number
    for i in range(neuron_number):
        neuron = 0
        for j in range(dictionary_length):
            neuron += input_vector[j] * weights[0][j][i]
        neuron = activation_function(neuron)
        hidden_layer[i] = neuron

    output_vector = [0] * 10
    for i in range(10):
        output = 0
        for j in range(neuron_number):
            output += hidden_layer[j] * weights[1][j][i]
        output = activation_function(output)
        output_vector[i] = output

    return output_vector


def calculate_accuracy(inputs, outputs, weights, dictionary_length = dl, neuron_number = neuron_number, activation_function = ReLU):
    TN_TP = 0
    TN_TP_FP_FN = 0

    for i in range(int(len(inputs)/50)):
        TN_TP_FP_FN += 1
        if classify(propogation(inputs[i*50], weights)) == classify(outputs[i*10]):
            TN_TP += 1

    return TN_TP/TN_TP_FP_FN


def crossed(wights1, weights2):
    w1 = deepcopy(wights1)
    w2 = deepcopy(weights2)

    w3 = deepcopy(wights1)

    for i in range(len(wights1)):
        for j in range(len(wights1[i])):
            for k in range(len(wights1[i][j])):
                if int((i+j+k)%2) == 1:
                    w3[i][j][k] = w1[i][j][k]
                else:
                    w3[i][j][k] = w2[i][j][k]
    return w3


def mutate(weights, percentage = .4, rate = .5, probabilty = .1):
    p = random.randint(0, 100) / 100
    if p < probabilty:
        return weights

    weights_unpacked = []
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights_unpacked.append(weights[i][j][k])
    
    for i in range(int(len(weights_unpacked) * percentage)):
        index = random.randrange(0, len(weights_unpacked))
        weights_unpacked[index] += random.randrange(int(-rate * 100), int(rate * 100)) / 100

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] = weights_unpacked[i+j+k]

    return weights


def train(input_vectors, outputs, population_count = 50, epochs = 1):

    global dl
    global neuron_number

    random_range = 2

    #generating
    population = [[] for i in range(population_count)]
    for i in range(population_count):
        population[i] = [[[0 for i in range(neuron_number)] for j in range(dl)], [[0 for i in range(10)] for j in range(neuron_number)]]
        for j in range(len(population[i])):
            for k in range(len(population[i][j])):
                for l in range(len(population[i][j][k])):
                    population[i][j][k][l] = random.randrange(-random_range * 100, random_range * 100) / 100

    for e in range(epochs):
        #crossing
        new_population = []
        for i in range(population_count):
            for j in range(i+1, population_count):
                new_population.extend([crossed(population[i], population[j])])

        #mutating
        for w in new_population:
            w = mutate(w)

        #selecting
        survived = []
        measured = []
        for j in new_population:
            measured.extend([calculate_accuracy(input_vectors, outputs, j)])
            print('N: {}/{}, ACC: {}, last: {}'.format(new_population.index(j), len(new_population), "%.3f" % max(measured),"%.3f" % measured[-1]))

        res_for_this_epoch = max(measured)

        for i in range(population_count):
            survived.extend([new_population[measured.index(max(measured))]])
            new_population.remove(new_population[measured.index(max(measured))])
            measured.remove(max(measured))

        population = survived[:]

        print('epoch {} finished with accuracy {}'.format(e, res_for_this_epoch))

#train([[[0,0,0],[0,0,0]], [[0],[0],[0]]], [[1, 2],[1, 3],[2, 2],[2, 3],[3, 3]], [3, 4, 4, 5, 6], epochs = 5)

train(input_vectors, outputs, epochs=100)