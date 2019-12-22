import codecs
import numpy as np
import random

categories = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']


dl = len(dictionary)
neuron_number = 100
weights = [[[0] * neuron_number] * dl, [[0] * 10] * neuron_number]
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

    for i in range(len(inputs)):
        TN_TP_FP_FN += 1
        if classify(propogation(inputs[i], weights)) == classify(outputs[i]):
            TN_TP += 1

    return TN_TP/TN_TP_FP_FN


def crossed(wights1, weights2):
    for i in range(len(wights1)):
        for j in range(len(wights1[i])):
            for k in range(len(wights1[i][j])):
                wights1[i][j][k] = (wights1[i][j][k] + weights2[i][j][k]) / 2
    return wights1


def mutate(weights, percentage = .2, rate = .1):
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


def train(weights, input_vectors, outputs, population_count = 10, epochs = 1):

    random_range = 2

    #generating
    population = []
    for i in range(population_count):
        w = weights[:]
        for j in range(len(w)):
            for k in range(len(w[j])):
                for l in range(len(w[j][k])):
                    w[j][k][l] = random.randrange(-random_range * 100, random_range * 100) / 100
        population.append(w)

    for e in range(epochs):
        #crossing
        new_population = []
        for i in range(population_count):
            for j in range(i+1, population_count):
                new_population.append(crossed(population[i], population[j]))

        #selecting
        survived = []
        for i in range(population_count):
            measured = []
            for j in new_population:
                measured.append(calculate_accuracy(input_vectors, outputs, j))
            print('ACC: {}'.format(max(measured)))
            survived.append(new_population[measured.index(max(measured))])
            new_population.remove(new_population[measured.index(max(measured))])

        #mutating
        for w in survived:
            w = mutate(w)

        population = survived[:]

        print('epoch {} finished with accuracy {}'.format(e, calculate_accuracy(input_vectors, outputs, survived[0])))

#train([[[0,0,0],[0,0,0]], [[0],[0],[0]]], [[1, 2],[1, 3],[2, 2],[2, 3],[3, 3]], [3, 4, 4, 5, 6], epochs = 5)

train(weights, input_vectors, outputs, epochs=5)