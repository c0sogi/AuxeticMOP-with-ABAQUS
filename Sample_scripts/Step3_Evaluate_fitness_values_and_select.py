import numpy as np
from dataclasses import asdict
from auxeticmop.PostProcessing import evaluate_all_fitness_values, selection
from auxeticmop.ParameterDefinitions import Parameters, fitness_definitions
from auxeticmop.FileIO import pickle_io

# Define fitness value evaluation version. Definitions are organized in auxeticmop.ParameterDefinitions.fitness_definitions
evaluation_version = 'ver3'

# Initialize parameters
parameters = Parameters(evaluation_version=evaluation_version)
parameters.post_initialize()

# Load topologies and results(field outputs from ABAQUS)
loaded_topologies = pickle_io('Topologies_1', mode='r')
parent_topologies = loaded_topologies['parent']
offspring_topologies = loaded_topologies['offspring']
parent_results = pickle_io('FieldOutput_1', mode='r')
offspring_results = pickle_io('FieldOutput_offspring_1', mode='r')
assert len(parent_topologies) == len(offspring_topologies) == len(parent_results) == len(offspring_results)

# Evaluate fitness values
parent_fitness_values = evaluate_all_fitness_values(fitness_definitions=fitness_definitions,
                                                    params_dict=asdict(parameters),
                                                    results=parent_results, topologies=parent_topologies)
offspring_fitness_values = evaluate_all_fitness_values(fitness_definitions=fitness_definitions,
                                                       params_dict=asdict(parameters),
                                                       results=offspring_results, topologies=offspring_topologies)
print('Shape of parent fitness values: ', parent_fitness_values.shape)
print('Shape of offspring fitness values: ', offspring_fitness_values.shape)

# Gather topologies, results, and fitness values
all_topologies = np.vstack((parent_topologies, offspring_topologies))
all_results = dict(parent_results | {entity_num + len(parent_results): offspring_results[entity_num]
                                     for entity_num in sorted(offspring_results.keys())})
all_fitness_values = evaluate_all_fitness_values(fitness_definitions=fitness_definitions, params_dict=asdict(parameters),
                                                 results=all_results, topologies=all_topologies)
print('Shape of all topologies: ', all_topologies.shape)
print('size of all results: ', len(all_results))
print('Shape of all fitness values: ', all_fitness_values.shape)


# Topologies of parent of next generation will be selected by pareto fronts criterion
pareto_indices = selection(all_fitness_values=all_fitness_values, selected_size=parameters.end_pop)
print('Entity numbers which are selected by pareto-front-criterion: \n', pareto_indices + 1)
selected_topologies = all_topologies[pareto_indices]
selected_results = {entity_num: all_results[pareto_idx + 1]
                    for entity_num, pareto_idx in enumerate(pareto_indices, start=1)}
print('Shape of selected topologies: ', selected_topologies.shape)
print('Size of selected results: ', len(selected_results))
pickle_io('Topologies_2', mode='w', to_dump={'parent': selected_topologies})
pickle_io('FieldOutput_2', mode='w', to_dump=selected_results)
