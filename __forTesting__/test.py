from auxeticmop import load_pickled_dict_data
from auxeticmop import fitness_definitions
from auxeticmop import Parameters, asdict, evaluate_all_fitness_values
import numpy as np

topo_parent = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\Topologies_1')['parent']
result_parent = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\FieldOutput_1')
topo_offspring = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\Topologies_1')['offspring']
result_offspring = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\FieldOutput_offspring_1')
params = Parameters()
params.post_initialize()

all_topologies = np.vstack((topo_parent, topo_offspring))
all_results = {key + len(result_parent): value for key, value in result_offspring.items()}
all_results.update(result_parent)
all_fitness_values = evaluate_all_fitness_values(
    vars_definitions=fitness_definitions[params.evaluation_version].vars_definitions,
    fitness_values_definitions=fitness_definitions[params.evaluation_version].fitness_definitions,
    params_dict=asdict(params), results=all_results, topologies=all_topologies)

print(all_fitness_values)