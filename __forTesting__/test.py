from auxeticmop import load_pickled_dict_data
from auxeticmop import fitness_value_definitions_ver3, fitness_evaluation_for_one_entity, var_and_definitions_ver3
from auxeticmop import Parameters, asdict

topo_parent = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\Topologies_1')['parent']
result_parent = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\FieldOutput_1')

assert len(topo_parent) == len(result_parent)
for entity_idx in range(len(topo_parent)):
    e = fitness_evaluation_for_one_entity(var_and_definitions=var_and_definitions_ver3,
                                          fitness_value_definitions=fitness_value_definitions_ver3,
                                          params=asdict(Parameters()),
                                          topology=topo_parent[entity_idx], result=result_parent[entity_idx+1])
    print(e)
