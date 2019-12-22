import random

dl = 5
neuron_number = 4
weights = [[[0 for i in range(neuron_number)] for j in range(dl)], [[0 for i in range(3)] for j in range(neuron_number)]]

population = [[] for i in range(3)]

population[0] = [1, 2, 3]

for i in range(3):
        population[i] = [[[0 for i in range(neuron_number)] for j in range(dl)], [[0 for i in range(3)] for j in range(neuron_number)]]
        for j in range(len(population[i])):
            for k in range(len(population[i][j])):
                for l in range(len(population[i][j][k])):
                    population[i][j][k][l] = random.randrange(-2 * 100, 2 * 100) / 100

print('hui')