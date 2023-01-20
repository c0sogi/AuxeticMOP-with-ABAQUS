import os
import numpy as np
from GraphicUserInterface import Parameters
from FileIO import offspring_import, parent_import
from PostProcessing import evaluate_fitness_values, selection
from GeneticAlgorithm import generate_offspring


def test_main():
    path = r'f:\shshsh\temp'
    gen = 1
    params = Parameters()
    new_selection = False
    os.chdir(path)

    # Import parent topologies and outputs of current generation
    topo_parent, result_parent = parent_import(gen_num=gen)

    topo_offspring, result_offspring = offspring_import(gen_num=gen)
    fitness_values_parent = evaluate_fitness_values(topo=topo_parent, result=result_parent, params=params)
    fitness_values_offspring = evaluate_fitness_values(topo=topo_offspring, result=result_offspring, params=params)
    fitness_values_parent_and_offspring = np.vstack((fitness_values_parent, fitness_values_offspring))

    # Topologies of parent of next generation will be selected by pareto fronts criterion
    _, next_generations = selection(all_topologies=np.vstack((topo_parent, topo_offspring)),
                                    all_fitness_values=fitness_values_parent_and_offspring,
                                    population_size=params.end_pop)
    print(len(next_generations))


def test_history(gen, path=None):
    from PostProcessing import open_history_output
    open_history_output(gen=gen, path=path)


if __name__ == '__main__':
    for i in range(10, 47):
        test_history(gen=i, path=r'F:\shshsh\data-23-1-4')
