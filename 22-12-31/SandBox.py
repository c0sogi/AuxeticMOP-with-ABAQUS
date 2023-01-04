from GeneticAlgorithm import random_parent_generation
import numpy as np

lx = 10
ly = 10
lz = 10
total_offsprings = 1
density = 0.01
mutation_probability = 0.05
add_probability = 0.01
timeout = 0.5

# random_parent_generation(lx, ly, lz, total_offsprings, density, mutation_probability, add_probability, timeout,
#                          save_file=False)

c = np.array([[-1, 1],
              [0.5, -0.5]])
c = np.absolute(c)

print(c)
print(np.argmax(c))